# third party
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ppo.storage import Batch, RolloutStorage


class PPO:
    def __init__(self,
                 agent,
                 clip_param,
                 ppo_epoch,
                 batch_size,
                 value_loss_coef,
                 entropy_coef,
                 learning_rate=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 unsupervised=False):

        self.unsupervised = unsupervised
        self.actor_critic = agent

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = batch_size

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(
            agent.parameters(), lr=learning_rate, eps=eps)
        self.reward_function = None

    def update(self, rollouts: RolloutStorage):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            sample: Batch
            for sample in data_generator:
                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, \
                _ = self.actor_critic.evaluate_actions(
                    sample.obs, sample.recurrent_hidden_states,
                    sample.masks, sample.actions)

                ratio = torch.exp(action_log_probs -
                                  sample.old_action_log_probs)
                surr1 = ratio * sample.adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * sample.adv
                action_loss = -torch.min(surr1, surr2).mean()

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
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()

                if self.unsupervised:
                    expected_return_delta = torch.mean(
                        rollouts.returns * torch.log(
                            action_log_probs / sample.old_action_log_probs))
                    rollouts.reward_params.grad = None
                    expected_return_delta.backward()

                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return dict(
            value_loss=value_loss_epoch,
            action_loss=action_loss_epoch,
            entropy=dist_entropy_epoch)
