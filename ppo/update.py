# stdlib
from collections import Counter
import itertools

# third party
import torch
import torch.nn as nn
import torch.optim as optim

# first party
from common.running_mean_std import RunningMeanStd
from ppo.storage import Batch, RolloutStorage


def f(x):
    x.sum().backward(retain_graph=True)


def global_norm(grads):
    norm = 0
    for grad in grads:
        norm += grad.norm(2)**2
    return norm**.5


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

        if self.unsupervised:
            self.unsupervised_optimizer = optim.Adam(
                gan.parameters(), lr=gan.learning_rate, eps=eps)

        self.optimizer = optim.Adam(
            actor_critic.parameters(), lr=learning_rate, eps=eps)

        self.gan = gan
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
        unsupervised_values = Counter()

        total_norm = torch.tensor(0, dtype=torch.float32)
        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.batch_size)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.batch_size)

            if self.unsupervised:
                generator = rollouts.feed_forward_generator(
                    advantages, self.batch_size)
                for sample in itertools.islice(generator, 0,
                                               self.gan.num_samples):
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
                        loss_components = self.compute_loss_components(batch)
                        grads = torch.autograd.grad(
                            self.compute_loss(
                                *loss_components, importance_weighting=None),
                            self.actor_critic.parameters())
                        sums[idxs] = batch.adv.mean()
                        probs[idxs] = dist.log_prob(goal).sum().exp()

                    weighted_gradients = torch.dot(sums, probs)
                    sq_gradients = torch.dot(sums, sums)
                    prediction_loss = torch.sum(
                        (probs - weighted_gradients / sq_gradients * sums)**2)
                    entropy_loss = -self.entropy_coef * entropies
                    one_hot = torch.zeros_like(dist.probs)
                    one_hot[0, -1] = 1
                    diff = (dist.probs - one_hot)**2
                    # unsupervised_loss = prediction_loss + entropy_loss
                    unsupervised_loss = diff.sum()
                    unsupervised_loss.mean().backward()
                    # gan_norm = global_norm(
                    #     [p.grad for p in self.gan.parameters()])
                    unsupervised_values.update(
                        unsupervised_loss=unsupervised_loss,
                        unweighted_norm=sums,
                        goal_log_prob=probs,
                        dist_mean=dist.mean.mean(),
                        dist_std=dist.stddev.mean(),
                        n=1,
                    )
                    # gan_norm=gan_norm)
                    nn.utils.clip_grad_norm_(self.gan.parameters(),
                                             self.max_grad_norm)
                    self.unsupervised_optimizer.step()
                    self.unsupervised_optimizer.zero_grad()
                self.gan.set_input(goal, sum(grad.sum() for grad in grads))

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
        if self.unsupervised:
            n = unsupervised_values.pop('n')
            for k, v in unsupervised_values.items():
                update_values[k] = torch.mean(v) / n

        return update_values
