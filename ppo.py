# third party
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rollouts import Batch, RolloutStorage


class PPO:
    def __init__(
        self,
        agent,
        clip_param,
        ppo_epoch,
        num_batch,
        value_loss_coef,
        learning_rate=None,
        eps=None,
        max_grad_norm=None,
        use_clipped_value_loss=True,
        aux_loss_only=False,
    ):

        self.aux_loss_only = aux_loss_only
        self.agent = agent

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_batch

        self.value_loss_coef = value_loss_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=eps)
        self.reward_function = None

    def update(self, rollouts: RolloutStorage):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        logger = collections.Counter()

        for e in range(self.ppo_epoch):
            if self.agent.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch
                )
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch
                )

            sample: Batch
            for sample in data_generator:
                # Reshape to do in a single forward pass for all steps
                act = self.agent(
                    inputs=sample.obs,
                    rnn_hxs=sample.recurrent_hidden_states,
                    masks=sample.masks,
                    action=sample.actions,
                )
                values = act.value
                action_log_probs = act.action_log_probs
                loss = act.aux_loss
                # log_values = act.log
                # logger.update(**log_values)

                if not self.aux_loss_only:
                    ratio = torch.exp(action_log_probs - sample.old_action_log_probs)
                    surr1 = ratio * sample.adv
                    surr2 = (
                        torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                        * sample.adv
                    )
                    action_loss = -torch.min(surr1, surr2).mean()
                    logger.update(action_loss=action_loss)
                    loss += action_loss

                if self.use_clipped_value_loss:

                    value_pred_clipped = sample.value_preds + (
                        values - sample.value_preds
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - sample.ret).pow(2)
                    value_losses_clipped = (value_pred_clipped - sample.ret).pow(2)
                    value_loss = (
                        0.5 * torch.max(value_losses, value_losses_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * F.mse_loss(sample.ret, values)
                logger.update(value_loss=value_loss)
                loss += self.value_loss_coef * value_loss

                self.optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # noinspection PyTypeChecker
                logger.update(n=1.0)

        n = logger.pop("n", 0)
        return {k: v.mean().item() / n for k, v in logger.items()}
