from enum import Enum

import numpy as np
import torch

from common.running_mean_std import RunningMeanStd
from ppo.util import Categorical, NoInput, init_normc_, mlp
from utils import ReplayBuffer, onehot

SamplingStrategy = Enum('SamplingStrategy',
                        'baseline pg gpg l2g gl2g abs_grads')


class TaskGenerator(NoInput):
    def __init__(self, task_size, temperature: float, exploration_bonus: float,
                 sampling_strategy):
        super().__init__(task_size)
        self.exploration_bonus = exploration_bonus
        self.sampling_strategy = sampling_strategy
        self.task_size = task_size
        self.temperature = temperature
        if sampling_strategy == SamplingStrategy.abs_grads.name:
            self._logits = torch.Tensor(1, task_size)
            init_normc_(self._logits)
            self._logits = self._logits.view(-1)
        else:
            self._logits = torch.ones(task_size)

    def logits(self):
        return self._logits

    def dist(self):
        return Categorical(logits=self.temperature * self.logits())

    def sample(self):
        return self.dist().sample()

    def probs(self):
        return self.dist().probs.detach().numpy()

    def importance_weight(self, probs):
        return 1 / (self.task_size * probs)

    def update(self, tasks, grads, step=None):
        if self.sampling_strategy != 'baseline':
            self._logits += self.exploration_bonus
            self._logits[tasks] = grads.cpu()


class RewardBasedTaskGenerator(TaskGenerator):
    def __init__(self, task_size, task_buffer_size, min_reward, max_reward,
                 **kwargs):
        super().__init__(task_size=task_size, **kwargs)
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.buffer_size = task_buffer_size
        self.histories = [
            ReplayBuffer(self.buffer_size) for _ in range(task_size)
        ]

    def dist(self):
        not_full = list(not h.full for h in self.histories)
        if any(not_full):
            not_full = torch.tensor(not_full)
            probs = not_full.float() / not_full.sum()
            return Categorical(probs=probs)
        if self.sampling_strategy == 'reward-variance':
            logits = [h.array().var() for h in self.histories]
            return Categorical(logits=self.temperature * torch.tensor(logits))
        if self.sampling_strategy == 'reward-range':
            rewards = torch.tensor([h.array().mean() for h in self.histories])
            in_range = (.1 < rewards) & (rewards < .9)
            if not torch.any(in_range):
                return Categorical(logits=torch.ones(self.task_size))
            in_range = in_range.float()
            return Categorical(probs=in_range / in_range.sum())

    def update(self, tasks, rewards, step=None):
        for task, reward in zip(tasks, rewards):
            normalized = (reward - self.min_reward) / (
                self.max_reward - self.min_reward)
            self.histories[task].append(normalized)


class GoalGAN(RewardBasedTaskGenerator):
    def __init__(self, size_noise, reward_lower_bound, reward_upper_bound, lrG,
                 lrD, writer, task_size, generator_network_args,
                 discriminator_network_args, **kwargs):
        super().__init__(task_size=task_size, **kwargs)
        self.time_step = 0
        self.reward_lower_bound = reward_lower_bound
        self.reward_upper_bound = reward_upper_bound
        self.noise_dist = torch.distributions.Normal(
            loc=torch.zeros(size_noise), scale=torch.ones(size_noise))
        self.G = mlp(
            num_inputs=size_noise, num_outputs=1, **generator_network_args)
        self.D = mlp(
            num_inputs=size_noise, num_outputs=1, **discriminator_network_args)
        self.noise = None
        self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=lrG)
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=lrD)
        self.writer = writer
        self.uniform = Categorical(logits=torch.ones(task_size))

    def sample(self):
        if self.time_step % 3 == 0:
            return self.uniform.sample()
        self.noise = self.noise_dist.sample()
        return self.G(self.noise).argmax()

    def update(self, tasks, rewards, step=None):
        super().update(tasks, rewards)
        self.time_step += 1
        task_vectors = onehot(tasks, self.task_size)
        task_vectors = torch.tensor(task_vectors)
        y = (rewards > self.reward_lower_bound) & (rewards <
                                                   self.reward_upper_bound)
        loss_d = y * (self.D(task_vectors) - 1)**2 + (1 - y) * (
            self.D(task_vectors) + 1)**2
        loss_d.mean().backward(retain_graph=True)
        self.D_optimizer.step()
        self.D_optimizer.zero_grad()
        loss_g = self.D(self.G(self.noise_dist.sample()))**2
        loss_g.mean().backward()
        self.G_optimizer.step()
        self.G_optimizer.zero_grad()
        if self.writer:
            self.writer.add_scalar('discriminator loss', loss_d, step)
            self.writer.add_scalar('generator loss', loss_g, step)
