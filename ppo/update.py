# stdlib
import math
from collections import Counter

# third party
import torch
import torch.nn as nn
import torch.optim as optim

# first party
from ppo.storage import RolloutStorage


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
                 delta,
                 learning_rate=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 goal_generator=None):

        self.train_goals = bool(goal_generator)
        self.actor_critic = actor_critic
        self.delta = delta

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

            for batch in data_generator:
                # Reshape to do in a single forward pass for all steps

                values, action_log_probs, dist_entropy, \
                _ = self.actor_critic.evaluate_actions(
                    batch.obs, batch.recurrent_hidden_states, batch.masks,
                    batch.actions)

                def log_prob_target_policy(alpha):
                    x = batch.adv * torch.log(alpha)
                    return batch.old_action_log_probs + x

                def KL(alpha):
                    return batch.old_action_log_probs - log_prob_target_policy(
                        alpha) - torch.log(torch.mean(alpha**batch.adv))
                    # return (1 + alpha**(-batch.ret)) * batch.ret * torch.log(alpha)

                def binary_search(alpha, diff, i):
                    kl = KL(alpha).mean()
                    if i == 0 or torch.abs(kl - self.delta) < .01:
                        return alpha, kl
                    if diff * (kl - self.delta) < 0:  # wrong direction
                        diff /= -2
                    return binary_search(alpha + diff, diff, i - 1)

                alpha, kl = binary_search(
                    torch.tensor(1.), torch.tensor(1.), 100)

                target = log_prob_target_policy(alpha)
                action_losses = (target - batch.old_action_log_probs).exp() * (
                    target - action_log_probs)

                value_losses = (values - batch.ret).pow(2)
                if self.use_clipped_value_loss:
                    value_pred_clipped = batch.value_preds + \
                                         (values - batch.value_preds).clamp(
                                             -self.clip_param, self.clip_param)
                value_losses_clipped = (value_pred_clipped - batch.ret).pow(2)
                value_losses = .5 * torch.max(value_losses,
                                              value_losses_clipped)

                loss = torch.mean(action_losses -
                                  dist_entropy * self.entropy_coef +
                                  value_losses * self.value_loss_coef)

                loss.backward()
                total_norm += global_norm(
                    [p.grad for p in self.actor_critic.parameters()])
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()
                # noinspection PyTypeChecker
                self.optimizer.zero_grad()
                update_values.update(
                    kl=kl,
                    value_loss=value_losses,
                    action_loss=action_losses,
                    norm=total_norm,
                    entropy=dist_entropy,
                    n=1)
                if batch.importance_weighting is not None:
                    update_values.update(
                        importance_weighting=batch.importance_weighting)

        n = update_values.pop('n')
        update_values = {
            k: torch.mean(v) / n
            for k, v in update_values.items()
        }
        if self.train_goals and 'n' in goal_values:
            n = goal_values.pop('n')
            for k, v in goal_values.items():
                update_values[k] = torch.mean(v) / n

        return update_values
