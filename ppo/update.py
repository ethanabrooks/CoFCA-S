# stdlib
# third party
# first party
from collections import Counter, namedtuple
from enum import Enum
import itertools
import math

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
                 learning_rate=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 task_generator=None):

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

        num_steps, num_processes = rollouts.rewards.size()[0:2]
        total_batch_size = num_steps * num_processes
        batches = rollouts.make_batch(advantages,
                                      torch.arange(total_batch_size))

        total_norm = torch.tensor(0, dtype=torch.float32)
        if self.train_tasks:
            tasks_trained = torch.unique(batches.tasks)
            task_grads = torch.zeros(tasks_trained.size()[0])
            task_returns = torch.zeros(tasks_trained.size()[0])

        for e in range(self.ppo_epoch):
            if self.train_tasks:
                _, action_losses, _ = self.compute_loss_components(
                    batches, compute_value_loss=False)
                for i, task in enumerate(tasks_trained):
                    action_loss = action_losses[batches.tasks == task]
                    task_returns[i] = torch.mean(
                        batches.ret[batches.tasks == task])
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
                        task_grads[i] = global_norm(
                            [p.grad for p in self.actor_critic.parameters()])
                    else:
                        task_grads[i] = sum(
                            g.abs().sum() for g in grad if g is not None)

                logits = next(self.task_generator.parameters()).view(-1)
                task_loss = torch.mean(
                    (logits[tasks_trained.long()] - task_grads)**2)
                task_loss.backward()
                self.task_optimizer.step()
                # TODO: task_optimizer.zero?
                update_values.update(task_loss=task_loss,
                                     grad_measure=task_grads,
                                     importance_weighting=batches.importance_weighting.mean()
                                     )

            # Compute loss
            value_losses, action_losses, entropy \
                = components = self.compute_loss_components(batches)
            loss = self.compute_loss(
                *components, importance_weighting=batches.importance_weighting)

            # update
            loss.backward()
            total_norm += global_norm(
                [p.grad for p in self.actor_critic.parameters()])
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

            update_values.update(
                value_loss=torch.mean(value_losses),
                action_loss=torch.mean(action_losses),
                norm=total_norm,
                entropy=torch.mean(entropy),
                n=1,)

        n = update_values.pop('n')
        update_values = {
            k: torch.mean(v) / n
            for k, v in update_values.items()
        }
        if self.train_tasks and 'n' in task_values:
            n = task_values.pop('n')
            for k, v in task_values.items():
                update_values[k] = torch.mean(v) / n

        if self.train_tasks:
            return update_values, (tasks_trained, task_returns, task_grads)
        else:
            return update_values, None
