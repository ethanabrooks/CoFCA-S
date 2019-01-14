# third party
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim

from common.running_mean_std import RunningMeanStd
from ppo.storage import Batch, RolloutStorage


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
            self.gradient_rms = RunningMeanStd()
        self.gan = gan
        self.reward_function = None

    def update(self, rollouts: RolloutStorage):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5)
        update_values = Counter()

        total_norm = torch.tensor(0, dtype=torch.float32)

        def compute_loss_components(batch):
            values, action_log_probs, dist_entropy, \
            _ = self.actor_critic.evaluate_actions(
                batch.obs, batch.recurrent_hidden_states, batch.masks,
                batch.actions)

            ratio = torch.exp(action_log_probs -
                              batch.old_action_log_probs)
            surr1 = ratio * batch.adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                1.0 + self.clip_param) * batch.adv

            _action_losses = -torch.min(surr1, surr2)

            _value_losses = (values - batch.ret).pow(2)
            if self.use_clipped_value_loss:
                value_pred_clipped = batch.value_preds + \
                                     (values - batch.value_preds).clamp(
                                         -self.clip_param, self.clip_param)
                value_losses_clipped = (
                        value_pred_clipped - batch.ret).pow(2)
                _value_losses = .5 * torch.max(_value_losses,
                                               value_losses_clipped)

            _importance_weighting = batch.importance_weighting
            if _importance_weighting is None:
                _importance_weighting = torch.tensor(
                    1, dtype=torch.float32)
            return (_value_losses, _action_losses, dist_entropy,
                    _importance_weighting)

        def compute_loss(value_loss, action_loss, dist_entropy,
                         _importance_weighting):
            if _importance_weighting is None:
                _importance_weighting = 1
            else:
                _importance_weighting = _importance_weighting.detach()
                _importance_weighting[torch.isnan(
                    _importance_weighting)] = 0
            losses = (value_loss * self.value_loss_coef + action_loss -
                      dist_entropy * self.entropy_coef)
            return torch.mean(losses * _importance_weighting)

        def global_norm(grads) -> torch.tensor:
            norm = 0
            for grad in grads:
                norm += grad.norm(2) ** 2
            return norm ** .5

        if self.unsupervised:
            sample = next(rollouts.feed_forward_generator(
                advantages, self.batch_size))
            self.unsupervised_optimizer.zero_grad()
            loss = compute_loss(*compute_loss_components(sample))
            grads = torch.autograd.grad(
                outputs=loss,
                inputs=self.actor_critic.parameters(),
                create_graph=True,
            )
            unsupervised_loss = -global_norm(grads)
            unsupervised_loss.mean().backward()
            # gan_norm = global_norm(
                # [p.grad for p in self.gan.parameters()])
            update_values.update(
                unsupervised_loss=unsupervised_loss,
                )
                # gan_norm=gan_norm)
            nn.utils.clip_grad_norm_(self.gan.parameters(),
                                     self.max_grad_norm)
            self.unsupervised_optimizer.step()

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.batch_size)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.batch_size)

            for sample in data_generator:
                # Reshape to do in a single forward pass for all steps
                self.optimizer.zero_grad()
                sample = sample._replace(obs=sample.obs.detach())
                value_losses, action_losses, entropy, importance_weighting \
                    = components = compute_loss_components(sample)
                loss = compute_loss(*components)
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
