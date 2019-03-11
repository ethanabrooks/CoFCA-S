# stdlib
# third party
# first party
from collections import Counter
from enum import Enum
import math

import torch
import torch.nn as nn
import torch.optim as optim

from ppo.storage import RolloutStorage, TasksRolloutStorage
from ppo.util import Categorical


def f(x):
    x.sum().backward(retain_graph=True)


def global_norm(grads):
    norm = 0
    for grad in grads:
        if grad is not None:
            norm += grad.norm(2) ** 2
    return norm ** .5


def epanechnikov_kernel(x):
    return 3 / 4 * (1 - x ** 2)


def gaussian_kernel(x):
    return (2 * math.pi) ** -.5 * torch.exp(-.5 * x ** 2)


SamplingStrategy = Enum(
    'SamplingStrategy',
    'baseline binary_logits gradients max learned learn_sampled')


class PPO:
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 batch_size,
                 value_loss_coef,
                 entropy_coef,
                 sampling_strategy,
                 exploration_bonus,
                 learning_rate=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 task_generator=None):

        self.exploration_bonus = exploration_bonus
        self.sampling_strategy = sampling_strategy
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

    def update(self, rollouts: RolloutStorage, importance_weighting):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        # advantages = (advantages - advantages.mean()) / (
        # advantages.std() + 1e-5)
        update_values = Counter()
        task_values = Counter()

        total_norm = torch.tensor(0, dtype=torch.float32)
        tasks_trained = []
        task_returns = []
        task_grads = []

        # compute_losses
        num_steps, num_processes = rollouts.rewards.size()[0:2]
        total_batch_size = num_steps * num_processes
        batches = rollouts.make_batch(advantages,
                                      torch.arange(total_batch_size))
        tasks_to_train = torch.unique(batches.tasks[batches.process == 0])
        grads = torch.zeros(tasks_to_train.size()[0])
        returns = torch.zeros(tasks_to_train.size()[0])

        for e in range(self.ppo_epoch):
            assert isinstance(rollouts, TasksRolloutStorage)

            for i, task in enumerate(tasks_to_train):
                uses_task = (batches.tasks == task).any(-1)
                train_indices = torch.arange(total_batch_size)[uses_task]
                batch = rollouts.make_batch(advantages, train_indices)
                _, action_loss, _ = self.compute_loss_components(
                    batch, compute_value_loss=False)

                returns[i] = torch.mean(batch.ret)
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
                grads[i] = sum(
                    g.abs().sum() for g in grad if g is not None)

            if self.sampling_strategy == SamplingStrategy.learn_sampled.name:
                logits = self.task_generator.logits
                logits += self.exploration_bonus
                logits[tasks_to_train.long()] = grads

            # update task data
            tasks_trained.extend(tasks_to_train)
            task_returns.extend(returns)
            task_grads.extend(grads)

            # make sample
            uses_task = (batches.process == 0).any(-1)
            train_indices = torch.arange(total_batch_size)[uses_task]
            sample = rollouts.make_batch(advantages, train_indices)

            # Compute loss
            value_losses, action_losses, entropy \
                = components = self.compute_loss_components(sample)
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

            update_values.update(grad_measure=grads,
                                 value_loss=torch.mean(value_losses),
                                 action_loss=torch.mean(action_losses),
                                 norm=total_norm,
                                 entropy=torch.mean(entropy),
                                 n=1)
            if importance_weighting is not None:
                update_values.update(
                    importance_weighting=importance_weighting.mean())

        return_values = {}

        def accumulate_values(counter):
            n = counter.pop('n')
            for k, v in counter.items():
                return_values[k] = torch.mean(v) / n

        accumulate_values(update_values)
        if self.train_tasks and 'n' in task_values:
            accumulate_values(task_values)

        return return_values, torch.tensor(
            tasks_trained), task_returns, task_grads
