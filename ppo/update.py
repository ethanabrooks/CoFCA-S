# third party
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim

from common.running_mean_std import RunningMeanStd
from ppo.storage import RolloutStorage, Batch


def f(x):
    x.sum().backward(retain_graph=True)


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
                 gan=None):

        self.unsupervised = bool(gan)
        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.batch_size = batch_size

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(
            actor_critic.parameters(), lr=learning_rate, eps=eps)

        if self.unsupervised:
            self.unsupervised_optimizer = optim.Adam(
                gan.parameters(), lr=gan.learning_rate, eps=eps)
            self.mean_weighted_gradient = RunningMeanStd()
            self.mean_sq_grad = RunningMeanStd()
        self.gan = gan
        self.reward_function = None

    def compute_loss_components(self, batch):
        values, action_log_probs, dist_entropy, \
        _ = self.actor_critic.evaluate_actions(
            batch.obs, batch.recurrent_hidden_states, batch.masks,
            batch.actions)

        ratio = torch.exp(action_log_probs -
                          batch.old_action_log_probs)
        surr1 = ratio * batch.adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                            1.0 + self.clip_param) * batch.adv

        action_losses = -torch.min(surr1, surr2)

        value_losses = (values - batch.ret).pow(2)
        if self.use_clipped_value_loss:
            value_pred_clipped = batch.value_preds + \
                                 (values - batch.value_preds).clamp(
                                     -self.clip_param, self.clip_param)
            value_losses_clipped = (
                    value_pred_clipped - batch.ret).pow(2)
            value_losses = .5 * torch.max(value_losses,
                                          value_losses_clipped)

        importance_weighting = batch.importance_weighting
        if importance_weighting is None:
            importance_weighting = torch.tensor(
                1, dtype=torch.float32)
        return (value_losses, action_losses, dist_entropy,
                importance_weighting)

    def compute_loss(self, value_loss, action_loss, dist_entropy,
                     importance_weighting):
        if importance_weighting is None:
            importance_weighting = 1
        else:
            importance_weighting = importance_weighting.detach()
            importance_weighting[torch.isnan(
                importance_weighting)] = 0
        losses = (value_loss * self.value_loss_coef + action_loss -
                  dist_entropy * self.entropy_coef)
        return torch.mean(losses * importance_weighting)

    def update(self, rollouts: RolloutStorage):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5)
        update_values = Counter()

        total_norm = torch.tensor(0, dtype=torch.float32)
        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.batch_size)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.batch_size)

            if self.unsupervised:
                self.unsupervised_optimizer.zero_grad()
                sample = next(rollouts.feed_forward_generator(
                    advantages, self.batch_size))
                unique = torch.unique(sample.goals, dim=0)
                probs = torch.zeros(len(sample.goals))
                sums = torch.zeros(len(sample.goals))
                entropies = torch.tensor(0.)
                indices = torch.arange(len(sample.goals))
                for goal in unique:
                    idxs = indices[(sample.goals == goal).all(dim=-1)]
                    dist = self.gan.dist(len(idxs))
                    entropies += dist.entropy().mean()
                    batch = Batch(*[x[idxs, ...] for x in sample])
                    *loss_components, _ = self.compute_loss_components(batch)
                    grads = torch.autograd.grad(
                        self.compute_loss(*loss_components,
                                          importance_weighting=None),
                        self.actor_critic.parameters())
                    global_sum = sum(grad.sum() for grad in grads)
                    prob = dist.log_prob(goal).sum(-1).exp()
                    sums[idxs] = global_sum
                    probs[idxs] = prob
                    weighted_gradient = (prob.detach() * global_sum)
                    self.mean_weighted_gradient.update(weighted_gradient.numpy(),
                                                       axis=None)
                    self.mean_sq_grad.update(global_sum.numpy() ** 2, axis=None)

                alpha = self.mean_weighted_gradient.mean / self.mean_sq_grad.mean
                unsupervised_loss = .5 * (probs - alpha * sums) ** 2 \
                                    + self.entropy_coef * entropies
                unsupervised_loss.mean().backward()
                # gan_norm = global_norm(
                #     [p.grad for p in self.gan.parameters()])
                update_values.update(
                    unsupervised_loss=unsupervised_loss,
                    goal_log_prob=probs.mean(),
                    dist_mean=dist.mean.mean(),
                    dist_std=dist.stddev.mean(),)
                    # gan_norm=gan_norm)
                nn.utils.clip_grad_norm_(self.gan.parameters(),
                                         self.max_grad_norm)
                self.unsupervised_optimizer.step()
                self.gan.set_input(goal, global_sum)

            for sample in data_generator:
                # Reshape to do in a single forward pass for all steps

                def global_norm(grads):
                    norm = 0
                    for grad in grads:
                        norm += grad.norm(2) ** 2
                    return norm ** .5

                self.optimizer.zero_grad()
                value_losses, action_losses, entropy, importance_weighting \
                    = components = self.compute_loss_components(sample)
                loss = self.compute_loss(*components)
                loss.backward()
                total_norm += global_norm(
                    [p.grad for p in self.actor_critic.parameters()])
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()
                # noinspection PyTypeChecker
                update_values.update(
                    value_loss=value_losses,
                    action_loss=action_losses,
                    norm=total_norm,
                    entropy=entropy,
                    importance_weighting=importance_weighting,
                )

        num_updates = self.ppo_epoch * self.batch_size
        return {
            k: v.mean().detach().numpy() / num_updates
            for k, v in update_values.items()
        }
