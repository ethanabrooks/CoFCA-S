# stdlib
# third party
# first party
import math
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim

from ppo.storage import RolloutStorage


def f(x):
    x.sum().backward(retain_graph=True)


def global_norm(grads):
    norm = 0
    for grad in grads:
        if grad is not None:
            norm += grad.norm(2) ** 2
    return norm ** .5


def l2_norm(list_of_tensors):
    return sum([torch.sum(x ** 2) for x in list_of_tensors if x is not None])


def epanechnikov_kernel(x):
    return 3 / 4 * (1 - x ** 2)


def gaussian_kernel(x):
    return (2 * math.pi) ** -.5 * torch.exp(-.5 * x ** 2)


class PPO:
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 batch_size,
                 value_loss_coef,
                 entropy_coef,
                 learning_rate=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 task_generator=None):

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
            self.sampling_strategy = self.task_generator.sampling_strategy

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

    def update(self, rollouts: RolloutStorage, gamma: float):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-5)
        update_values = Counter()
        task_values = Counter()

        total_norm = torch.tensor(0, dtype=torch.float32)

        # compute_losses
        num_steps, num_processes = rollouts.rewards.size()[0:2]
        total_batch_size = num_steps * num_processes
        batches = rollouts.make_batch(advantages,
                                      torch.arange(total_batch_size))

        if self.train_tasks:
            tasks_to_train = torch.unique(batches.tasks)

            if self.sampling_strategy == 'l2g':
                pre_update_l2 = l2_norm(self.actor_critic.parameters())
            if self.sampling_strategy == 'pg':
                pre_update_loss = torch.zeros_like(tasks_to_train, dtype=torch.float)
                for i, task in enumerate(tasks_to_train):
                    uses_task = (batches.tasks == task).any(-1)
                    train_indices = torch.arange(total_batch_size)[uses_task]
                    batch = rollouts.make_batch(advantages, train_indices)
                    _, action_loss, _ = self.compute_loss_components(
                        batch, compute_value_loss=False)

                    loss = self.compute_loss(
                        action_loss=action_loss,
                        dist_entropy=0,
                        value_loss=None,
                        importance_weighting=None)

                    pre_update_loss[i] = loss

        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                advantages, self.batch_size)

            for sample in data_generator:
                # Compute loss
                value_losses, action_losses, entropy \
                    = components = self.compute_loss_components(sample)
                if self.train_tasks:
                    exp = self.task_generator.task_size - sample.tasks.float() - 1
                    entropy *= torch.abs(sample.value_preds - gamma ** exp)
                loss = self.compute_loss(
                    value_losses, action_losses, entropy,
                    importance_weighting=sample.importance_weighting)

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
                    n=1)
                if sample.importance_weighting is not None:
                    update_values.update(
                        importance_weighting=sample.importance_weighting.mean())

        if self.train_tasks:
            grads_per_step = torch.zeros(total_batch_size)
            grads_per_task = torch.zeros_like(
                tasks_to_train, dtype=torch.float)
            post_update_loss = torch.zeros_like(
                tasks_to_train, dtype=torch.float)
            grad_l2 = torch.zeros_like(
                tasks_to_train, dtype=torch.float)
            grads_list = []

            for i, task in enumerate(tasks_to_train):
                uses_task = (batches.tasks == task).any(-1)
                train_indices = torch.arange(total_batch_size)[uses_task]
                batch = rollouts.make_batch(advantages, train_indices)
                _, action_loss, _ = self.compute_loss_components(
                    batch, compute_value_loss=False)

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
                if self.sampling_strategy == 'gl2g':
                    grads_list.append(grad)
                if self.sampling_strategy == 'gpg':
                    grad_l2[i] = l2_norm(grad)
                post_update_loss[i] = loss
                grads_per_task[i] = grads_per_step[uses_task] = sum(
                    g.abs().sum() for g in grad if g is not None)

            if self.task_generator.sampling_strategy == 'abs_grads':
                self.task_generator.update(tasks_to_train, grads_per_task)
            elif self.task_generator.sampling_strategy == 'pg':
                self.task_generator.update(tasks_to_train,
                                           post_update_loss - pre_update_loss)
            elif self.task_generator.sampling_strategy == 'gpg':
                self.task_generator.update(tasks_to_train, grad_l2)
            elif self.task_generator.sampling_strategy == 'l2g':
                post_update_l2 = l2_norm(self.actor_critic.parameters())
                self.task_generator.update(tasks_to_train, post_update_l2 - pre_update_l2)
            elif self.task_generator.sampling_strategy == 'gl2g':
                dot_product = []
                parameters = self.actor_critic.parameters()
                grads_list = zip(*grads_list)
                for p, g in zip(parameters, grads_list):
                    if all(x is not None for x in g):
                        dot_product.append(torch.sum(p * sum(g) / len(g)))

                self.task_generator.update(tasks_to_train, sum(dot_product))

            task_values.update(grad_measure=grads_per_step, n=1)

        return_values = {}

        def accumulate_values(counter):
            n = counter.pop('n')
            for k, v in counter.items():
                return_values[k] = torch.mean(v) / n

        accumulate_values(update_values)
        if self.train_tasks and 'n' in task_values:
            accumulate_values(task_values)

        if self.train_tasks:
            return return_values, tasks_to_train, grads_per_task
        else:
            return return_values, None
