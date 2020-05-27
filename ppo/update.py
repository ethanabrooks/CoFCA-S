# third party
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ppo.storage import Batch, RolloutStorage


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
        def get_advantages(value_preds):
            a = rollouts.returns[:-1] - value_preds[:-1]
            if a.numel() > 1:
                return (a - a.mean()) / (a.std() + 1e-5)
            return a

        advantages = (
            get_advantages(rollouts.value_preds),
            get_advantages(rollouts.value_preds2),
            get_advantages(rollouts.value_preds3),
        )

        logger = collections.Counter()

        for e in range(self.ppo_epoch):
            if self.agent.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    *advantages, self.num_mini_batch
                )
            else:
                data_generator = rollouts.feed_forward_generator(
                    *advantages, self.num_mini_batch
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
                loss = act.aux_loss
                # log_values = act.log
                # logger.update(**log_values)

                if not self.aux_loss_only:

                    def get_action_loss(log_probs, old_log_probs, adv):
                        ratio = torch.exp(log_probs - old_log_probs)
                        surr1 = ratio * adv
                        surr2 = (
                            torch.clamp(
                                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                            )
                            * adv
                        )
                        return -torch.min(surr1, surr2).mean()

                    for lp, olp, a in [
                        (act.action_log_probs, sample.old_action_log_probs, sample.adv),
                        (
                            act.action_log_probs2,
                            sample.old_action_log_probs2,
                            sample.adv2,
                        ),
                        (
                            act.action_log_probs3,
                            sample.old_action_log_probs3,
                            sample.adv3,
                        ),
                    ]:
                        action_loss = get_action_loss(lp, olp, a)
                        logger.update(action_loss=action_loss)
                        loss += action_loss

                def get_value_loss(values, value_preds):
                    if self.use_clipped_value_loss:

                        value_pred_clipped = value_preds + (values - value_preds).clamp(
                            -self.clip_param, self.clip_param
                        )
                        value_losses = (values - sample.ret).pow(2)
                        value_losses_clipped = (value_pred_clipped - sample.ret).pow(2)
                        return (
                            0.5 * torch.max(value_losses, value_losses_clipped).mean()
                        )
                    else:
                        return 0.5 * F.mse_loss(sample.ret, values)

                for v, vp in [
                    (act.value, sample.value_preds),
                    (act.value2, sample.value_preds2),
                    (act.value3, sample.value_preds3),
                ]:
                    value_loss = get_value_loss(v, vp)
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
