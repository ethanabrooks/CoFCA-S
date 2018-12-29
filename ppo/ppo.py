# third party
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ppo.storage import RolloutStorage


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
        self.num_mini_batch = batch_size

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(
            actor_critic.parameters(), lr=learning_rate, eps=eps)

        if self.unsupervised:
            self.unsupervised_optimizer = optim.Adam(
                gan.parameters(), lr=learning_rate, eps=eps)
        self.reward_function = None

    def update(self, rollouts: RolloutStorage):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        update_values = Counter()

        total_norm = 0
        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, \
                _ = self.actor_critic.evaluate_actions(
                    sample.obs, sample.recurrent_hidden_states, sample.masks,
                    sample.actions)

                def compute_action_loss(J):
                    ratio = torch.exp(action_log_probs -
                                      sample.old_action_log_probs)
                    surr1 = ratio * J
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                        1.0 + self.clip_param) * J
                    return -torch.min(surr1, surr2).mean()

                action_loss = compute_action_loss(sample.adv)

                if self.use_clipped_value_loss:

                    value_pred_clipped = sample.value_preds + \
                                         (values - sample.value_preds).clamp(
                                             -self.clip_param, self.clip_param)
                    value_losses = (values - sample.ret).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - sample.ret).pow(2)
                    value_loss = .5 * torch.max(value_losses,
                                                value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * F.mse_loss(sample.ret, values)

                self.optimizer.zero_grad()
                loss = (value_loss * self.value_loss_coef + action_loss -
                        dist_entropy * self.entropy_coef)

                def global_norm(grads):
                    norm = 0
                    for grad in grads:
                        norm += grad.norm(2)**2
                    return norm**.5

                if self.unsupervised:
                    grads = torch.autograd.grad(
                        compute_action_loss(sample.ret),
                        self.actor_critic.base.actor.parameters(),
                        create_graph=True)
                    unsupervised_loss = global_norm(grads)
                    unsupervised_loss.backward(retain_graph=True)

                    update_values.update(unsupervised_loss=unsupervised_loss.
                                         squeeze().detach().numpy())
                    self.unsupervised_optimizer.step()

                loss.backward(retain_graph=True)
                total_norm += global_norm(
                    [p.grad for p in self.actor_critic.parameters()])
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()
                update_values.update(
                    value_loss=value_loss.detach().numpy(),
                    action_loss=action_loss.detach().numpy(),
                    entropy=dist_entropy.detach().numpy(),
                    norm=total_norm.detach().numpy())

        num_updates = self.ppo_epoch * self.num_mini_batch
        return {k: v / num_updates for k, v in update_values.items()}
