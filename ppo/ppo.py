# third party
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim

from common.running_mean_std import RunningMeanStd
from ppo.storage import RolloutStorage


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
        self.num_mini_batch = batch_size

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
        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                # Reshape to do in a single forward pass for all steps
                def compute_loss_components():
                    values, action_log_probs, dist_entropy, \
                    _ = self.actor_critic.evaluate_actions(
                        sample.obs, sample.recurrent_hidden_states, sample.masks,
                        sample.actions)

                    ratio = torch.exp(action_log_probs -
                                      sample.old_action_log_probs)
                    surr1 = ratio * sample.adv
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                        1.0 + self.clip_param) * sample.adv

                    action_losses = -torch.min(surr1, surr2)

                    value_losses = (values - sample.ret).pow(2)
                    if self.use_clipped_value_loss:
                        value_pred_clipped = sample.value_preds + \
                                             (values - sample.value_preds).clamp(
                                                 -self.clip_param, self.clip_param)
                        value_losses_clipped = (
                            value_pred_clipped - sample.ret).pow(2)
                        value_losses = .5 * torch.max(value_losses,
                                                      value_losses_clipped)

                    return value_losses, action_losses, dist_entropy

                def compute_loss(value_loss, action_loss, dist_entropy):
                    if sample.importance_weighting is None:
                        importance_weighting = 1
                    else:
                        importance_weighting = sample.importance_weighting.detach(
                        )
                        importance_weighting[torch.isnan(
                            importance_weighting)] = 0
                    losses = (value_loss * self.value_loss_coef + action_loss -
                              dist_entropy * self.entropy_coef)
                    return torch.mean(losses * importance_weighting)

                def global_norm(grads):
                    norm = 0
                    for grad in grads:
                        norm += grad.norm(2)**2
                    return norm**.5

                if self.unsupervised:
                    grads = torch.autograd.grad(
                        compute_loss(*compute_loss_components()),
                        self.actor_critic.parameters())
                    norm = global_norm(grads).detach()
                    self.gradient_rms.update(norm.numpy(), axis=None)
                    dist = self.gan.dist(sample.samples.size()[0])
                    log_prob = dist.log_prob(sample.samples)

                    unsupervised_loss = -log_prob * (
                            norm - self.gradient_rms.mean) - (
                            dist.entropy() * self.gan.entropy_coef)
                    unsupervised_loss.mean().backward()
                    gan_norm = global_norm(
                        [p.grad for p in self.gan.parameters()])
                    update_values.update(
                        unsupervised_loss=unsupervised_loss,
                        goal_log_prob=log_prob,
                        gan_norm=gan_norm)
                    self.unsupervised_optimizer.step()
                    self.unsupervised_optimizer.zero_grad()
                self.optimizer.zero_grad()
                value_losses, action_losses, entropy = components \
                    = compute_loss_components()
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
                )

        num_updates = self.ppo_epoch * self.num_mini_batch
        return {
            k: v.mean().detach().numpy() / num_updates
            for k, v in update_values.items()
        }
