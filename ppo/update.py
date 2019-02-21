# stdlib
from collections import Counter
import itertools
import math

# third party
# first party
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
        norm += grad.norm(2)**2
    return norm**.5


def epanechnikov_kernel(x):
    return 3 / 4 * (1 - x**2)


def gaussian_kernel(x):
    return (2 * math.pi)**-.5 * torch.exp(-.5 * x**2)


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
                 use_value,
                 learning_rate=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 task_generator=None):

        self.use_value = use_value
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

        self.optimizer = optim.Adam(
            actor_critic.parameters(), lr=learning_rate, eps=eps)

        self.gan = task_generator
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
        rets = []
        grad_sums = []
        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.batch_size)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.batch_size)

            assert isinstance(rollouts, TasksRolloutStorage)

            num_steps, num_processes = rollouts.rewards.size()[0:2]
            total_batch_size = num_steps * num_processes
            batches = rollouts.make_batch(advantages,
                                          torch.arange(total_batch_size))
            _, action_losses, _ = self.compute_loss_components(
                batches, compute_value_loss=False)
            unique = torch.unique(batches.tasks)
            grads = torch.zeros(unique.size()[0])
            for i, task in enumerate(unique):
                action_loss = action_losses[batches.tasks == task]
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
                grads[i] = sum(g.abs().sum() for g in grad if g is not None)

            if self.sampling_strategy == 'baseline':
                logits = torch.ones_like(grads)
            elif self.sampling_strategy == '0/1logits':
                logits = torch.ones_like(grads) * -self.temperature
                sorted_grads, _ = torch.sort(grads)
                mid_grad = sorted_grads[grads.numel() // 2]
                logits[grads > mid_grad] = self.temperature
            elif self.sampling_strategy == 'experiment':
                logits = grads * self.temperature
            elif self.sampling_strategy == 'max':
                logits = torch.ones_like(grads) * -self.temperature
                logits[grads.argmax()] = self.temperature
            else:
                raise RuntimeError

            dist = Categorical(logits=logits)
            task_index = dist.sample().long()
            task_to_train = unique[task_index]
            tasks_trained.append(task_to_train)

            importance_weighting = 1 / (
                unique.numel() * dist.log_prob(task_index).exp())

            uses_task = batches.tasks.squeeze() == task_to_train
            ret = batches.ret[uses_task].mean()
            grad = grads[unique == task_to_train]
            rets.append(ret)
            grad_sums.append(grad)
            # uses_task = torch.from_numpy(
            # np.isin(batches.tasks.numpy(),
            # [0, 1, 2, 8, 9, 10]).astype(np.uint8)).squeeze()
            indices = torch.arange(total_batch_size)[uses_task]
            sample = rollouts.make_batch(advantages, indices)

            # Reshape to do in a single forward pass for all steps

            value_losses, action_losses, entropy \
                = components = self.compute_loss_components(sample)
            loss = self.compute_loss(
                *components, importance_weighting=importance_weighting)
            loss.backward()
            total_norm += global_norm(
                [p.grad for p in self.actor_critic.parameters()])
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)
            self.optimizer.step()
            # noinspection PyTypeChecker
            self.optimizer.zero_grad()
            update_values.update(
                dist_mean=dist.mean,
                dist_std=dist.stddev,
                grad_sum=grads,
                value_loss=value_losses,
                action_loss=action_losses,
                norm=total_norm,
                entropy=entropy,
                task_trained=task_to_train,
                n=1)
            if importance_weighting is not None:
                update_values.update(importance_weighting=importance_weighting)

        n = update_values.pop('n')
        update_values = {
            k: torch.mean(v) / n
            for k, v in update_values.items()
        }
        if self.train_tasks and 'n' in task_values:
            n = task_values.pop('n')
            for k, v in task_values.items():
                update_values[k] = torch.mean(v) / n

        return update_values, tasks_trained, rets, grad_sums
