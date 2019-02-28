# stdlib
from collections import Counter, namedtuple
import itertools
import math

# third party
# first party
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from common.running_mean_std import RunningMeanStd
from ppo.storage import Batch, RolloutStorage, TasksRolloutStorage
from ppo.util import Categorical


def f(x):
    x.sum().backward(retain_graph=True)


def global_norm(grads):
    norm = 0
    for grad in grads:
        if grad is not None:
            norm += grad.norm(2)**2
    return norm**.5


def epanechnikov_kernel(x):
    return 3 / 4 * (1 - x**2)


def gaussian_kernel(x):
    return (2 * math.pi)**-.5 * torch.exp(-.5 * x**2)


SamplingStrategy = Enum(
    'SamplingStrategy', 'baseline binary_logits gradients max '
    'learned learn_sampled')


class PPO:
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 batch_size,
                 value_loss_coef,
                 entropy_coef,
                 temperature,
                 sampling_strategy,
                 global_norm,
                 num_processes,
                 learning_rate=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 task_generator=None):

        self.num_processes = torch.tensor(num_processes, dtype=torch.long)
        self.global_norm = global_norm
        self.sampling_strategy = sampling_strategy
        self.temperature = temperature
        self.train_tasks = bool(task_generator)
        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.batch_size = batch_size

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        if self.train_tasks:
            self.task_optimizer = optim.Adam(
                task_generator.parameters(),
                lr=task_generator.learning_rate,
                eps=eps)
            self.task_generator = task_generator

        self.optimizer = optim.Adam(
            actor_critic.parameters(), lr=learning_rate, eps=eps)

        self.reward_function = None

    def compute_loss_components(self, batch, compute_value_loss=True):
        values, action_log_probs, dist_entropy, \
        _ = self.actor_critic.evaluate_actions(
            batch.obs, batch.recurrent_hidden_states, batch.masks,
            batch.actions)

        ratio = torch.exp(action_log_probs - batch.old_action_log_probs)
        surr1 = ratio * batch.adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                            1.0 + self.clip_param) * batch.adv

        action_losses = -torch.min(surr1, surr2)

        value_losses = None
        if compute_value_loss:
            value_losses = (values - batch.ret).pow(2)
            if self.use_clipped_value_loss:
                value_pred_clipped = batch.value_preds + \
                                     (values - batch.value_preds).clamp(
                                         -self.clip_param, self.clip_param)
                value_losses_clipped = (value_pred_clipped - batch.ret).pow(2)
                value_losses = .5 * torch.max(value_losses,
                                              value_losses_clipped)

        return value_losses, action_losses, dist_entropy

    def compute_loss(self, value_loss, action_loss, dist_entropy,
                     importance_weighting):
        losses = (action_loss - dist_entropy * self.entropy_coef)
        if value_loss is not None:
            losses += value_loss * self.value_loss_coef

        if importance_weighting is not None:
            importance_weighting = importance_weighting.detach()
            importance_weighting[torch.isnan(importance_weighting)] = 0
            losses *= importance_weighting
        return torch.mean(losses)

    def update(self, rollouts: RolloutStorage):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5)
        update_values = Counter()
        task_values = Counter()

        total_norm = torch.tensor(0, dtype=torch.float32)
        tasks_trained = []
        task_returns = []
        task_grads = []

        for e in range(self.ppo_epoch):
            assert isinstance(rollouts, TasksRolloutStorage)

            num_steps, num_processes = rollouts.rewards.size()[0:2]
            total_batch_size = num_steps * num_processes
            batches = rollouts.make_batch(advantages,
                                          torch.arange(total_batch_size))
            _, action_losses, _ = self.compute_loss_components(
                batches, compute_value_loss=False)
            unique = torch.unique(batches.tasks)
            grads = torch.zeros(unique.size()[0])
            returns = torch.zeros(unique.size()[0])
            for i, task in enumerate(unique):
                action_loss = action_losses[batches.tasks == task]
                returns[i] = torch.mean(batches.ret[batches.tasks == task])
                loss = self.compute_loss(
                    action_loss=action_loss,
                    dist_entropy=0,
                    value_loss=None,
                    importance_weighting=None)
                grad = torch.autograd.grad(
                    loss,
                    self.actor_critic.parameters(),
                    retain_graph=True,
                    allow_unused=True)
                if self.global_norm:
                    grads[i] = global_norm(
                        [p.grad for p in self.actor_critic.parameters()])
                else:
                    grads[i] = sum(
                        g.abs().sum() for g in grad if g is not None)

            if self.sampling_strategy == SamplingStrategy.baseline.name:
                logits = torch.ones_like(grads)
            elif self.sampling_strategy == SamplingStrategy.binary_logits.name:
                logits = torch.ones_like(grads) * -self.temperature
                sorted_grads, _ = torch.sort(grads)
                mid_grad = sorted_grads[grads.numel() // 4]
                logits[grads > mid_grad] = self.temperature
            elif self.sampling_strategy == SamplingStrategy.gradients.name:
                logits = grads * self.temperature
            elif self.sampling_strategy == SamplingStrategy.max.name:
                logits = torch.ones_like(grads) * -self.temperature
                logits[grads.argmax()] = self.temperature
            elif self.sampling_strategy in [SamplingStrategy.learned.name,
                                            SamplingStrategy.learn_sampled.name]:
                logits = self.task_generator.parameter
            else:
                raise RuntimeError

            # sample tasks
            dist = Categorical(logits=logits.repeat(self.num_processes, 1))
            task_indices = dist.sample().view(-1).long()
            tasks_to_train = unique[task_indices]

            def update_task_params(logits_to_update, targets):
                task_loss = torch.mean((logits_to_update - targets) ** 2)
                task_loss.backward()
                self.task_optimizer.step()
                mean_abs_task_error = torch.mean(torch.abs(logits - grads))
                update_values.update(task_loss=task_loss,
                                     mean_abs_task_error=mean_abs_task_error)

            if self.sampling_strategy == SamplingStrategy.learned.name:
                update_task_params(logits, grads)
            elif self.sampling_strategy == SamplingStrategy.learn_sampled.name:
                update_task_params(logits[task_indices], grads[task_indices])

            # update task data
            tasks_trained.extend(tasks_to_train)
            task_returns.extend(returns[task_indices])
            task_grads.extend(grads[task_indices])

            # make sample
            uses_task = (batches.tasks == tasks_to_train).any(-1)
            train_indices = torch.arange(total_batch_size)[uses_task]
            sample = rollouts.make_batch(advantages, train_indices)

            # Compute loss
            value_losses, action_losses, entropy \
                = components = self.compute_loss_components(sample)
            task_to_train_index = torch.argmax(sample.tasks == tasks_to_train, dim=-1,
                                               keepdim=True)
            probs = dist.log_prob(task_indices).exp()[task_to_train_index]
            importance_weighting = 1 / (unique.numel() * probs)
            loss = self.compute_loss(
                *components, importance_weighting=importance_weighting)

            # update
            loss.backward()
            total_norm += global_norm(
                [p.grad for p in self.actor_critic.parameters()])
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

            update_values.update(grad_measure=grads)
            update_values.update(value_loss=torch.mean(value_losses))
            update_values.update(action_loss=torch.mean(action_losses))
            update_values.update(norm=total_norm)
            update_values.update(entropy=torch.mean(entropy))
            update_values.update(task_trained=tasks_to_train)
            update_values.update(n=1)
            # update_values.update(
            # dist_mean=dist.mean,
            # dist_std=dist.stddev,
            # grad_measure=grads,
            # value_loss=value_losses,
            # action_loss=action_losses,
            # norm=total_norm,
            # entropy=entropy,
            # task_trained=tasks_to_train,
            # n=1)
            if importance_weighting is not None:
                update_values.update(importance_weighting=importance_weighting.mean())

        n = update_values.pop('n')
        update_values = {
            k: torch.mean(v) / n
            for k, v in update_values.items()
        }
        if self.train_tasks and 'n' in task_values:
            n = task_values.pop('n')
            for k, v in task_values.items():
                update_values[k] = torch.mean(v) / n

        return update_values, torch.tensor(tasks_trained), task_returns, task_grads
