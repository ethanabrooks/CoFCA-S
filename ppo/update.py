# stdlib
from collections import Counter
import itertools
import math

# third party
import torch
import torch.nn as nn
import torch.optim as optim

# first party
from common.running_mean_std import RunningMeanStd
from ppo.storage import Batch, GoalsRolloutStorage, RolloutStorage


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
                 learning_rate=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 goal_generator=None):

        self.train_goals = bool(goal_generator)
        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.batch_size = batch_size

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        if self.train_goals:
            self.goal_optimizer = optim.Adam(
                goal_generator.parameters(),
                lr=goal_generator.learning_rate,
                eps=eps)

        self.optimizer = optim.Adam(
            actor_critic.parameters(), lr=learning_rate, eps=eps)

        self.gan = goal_generator
        self.reward_function = None

    def compute_loss_components(self, batch):
        values, action_log_probs, dist_entropy, \
        _ = self.actor_critic.evaluate_actions(
            batch.obs, batch.recurrent_hidden_states, batch.masks,
            batch.actions)

        ratio = torch.exp(action_log_probs - batch.old_action_log_probs)
        surr1 = ratio * batch.adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                            1.0 + self.clip_param) * batch.adv

        action_losses = -torch.min(surr1, surr2)

        value_losses = (values - batch.ret).pow(2)
        if self.use_clipped_value_loss:
            value_pred_clipped = batch.value_preds + \
                                 (values - batch.value_preds).clamp(
                                     -self.clip_param, self.clip_param)
            value_losses_clipped = (value_pred_clipped - batch.ret).pow(2)
            value_losses = .5 * torch.max(value_losses, value_losses_clipped)

        return value_losses, action_losses, dist_entropy

    def compute_loss(self, value_loss, action_loss, dist_entropy,
                     importance_weighting):
        losses = (value_loss * self.value_loss_coef + action_loss -
                  dist_entropy * self.entropy_coef)
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
        goal_values = Counter()

        total_norm = torch.tensor(0, dtype=torch.float32)
        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.batch_size)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.batch_size)

            if self.train_goals:
                assert isinstance(rollouts, GoalsRolloutStorage)

                goals, logits = rollouts.get_goal_batch(advantages)
                dist = self.gan.dist(1)
                probs = dist.log_prob(goals).exp()

                target = 1 / (self.gan.goal_size * logits.mean()) * logits
                true_target = 2 * logits / (self.gan.goal_size *
                                            (self.gan.goal_size - 1))

                diff = (probs - target)**2
                # goals_loss = prediction_loss + entropy_loss
                goal_loss = diff.mean()
                goal_loss.mean().backward()
                # gan_norm = global_norm(
                #     [p.grad for p in self.gan.parameters()])
                kernel_density_mse = torch.mean((target - true_target)**2)
                goal_values.update(
                    goal_loss=goal_loss,
                    kernel_density_mse=kernel_density_mse,
                    n=1)
                # gan_norm=gan_norm)
                nn.utils.clip_grad_norm_(self.gan.parameters(),
                                         self.max_grad_norm)
                self.goal_optimizer.step()
                self.goal_optimizer.zero_grad()
                # self.gan.set_input(goal, sum(grad.sum() for grad in grads))

            for sample in data_generator:
                # Reshape to do in a single forward pass for all steps

                value_losses, action_losses, entropy \
                    = components = self.compute_loss_components(sample)
                loss = self.compute_loss(
                    *components,
                    importance_weighting=sample.importance_weighting)
                loss.backward()
                total_norm += global_norm(
                    [p.grad for p in self.actor_critic.parameters()])
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                # self.optimizer.step()
                # noinspection PyTypeChecker
                self.optimizer.zero_grad()
                update_values.update(
                    value_loss=value_losses,
                    action_loss=action_losses,
                    norm=total_norm,
                    entropy=entropy,
                    n=1)
                if sample.importance_weighting is not None:
                    update_values.update(
                        importance_weighting=sample.importance_weighting)

        n = update_values.pop('n')
        update_values = {
            k: torch.mean(v) / n
            for k, v in update_values.items()
        }
        if self.train_goals:
            n = goal_values.pop('n')
            for k, v in goal_values.items():
                update_values[k] = torch.mean(v) / n

        return update_values
