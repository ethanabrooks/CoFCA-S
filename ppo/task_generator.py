from enum import Enum

import numpy as np
import torch

from common.running_mean_std import RunningMeanStd
from ppo.util import Categorical, NoInput, init_normc_, mlp
from rl_utils import ReplayBuffer, onehot

SamplingStrategy = Enum('SamplingStrategy',
                        'uniform pg gpg l2g gl2g abs_grads mtl')


class TaskGenerator(NoInput):
    def __init__(self, task_size, temperature: float, exploration_bonus: float,
                 sampling_strategy):
        super().__init__(task_size)
        self.exploration_bonus = exploration_bonus
        self.sampling_strategy = sampling_strategy
        self.task_size = task_size
        self.temperature = temperature
        if self.sampling_strategy == 'uniform':
            self.temperature = 1
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

    def update(self, grads, step=None):
        if self.sampling_strategy != 'uniform':
            self._logits += self.exploration_bonus
            for k, v in grads.items():
                self._logits[k] = v


class RewardBasedTaskGenerator(TaskGenerator):
    def __init__(self, task_size, task_buffer_size, reward_bounds, **kwargs):
        super().__init__(task_size=task_size, **kwargs)
        self.reward_lower_bound, self.reward_upper_bound = None, None
        if self.sampling_strategy == 'reward-range':
            self.reward_lower_bound, self.reward_upper_bound = reward_bounds
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
            in_range = (self.reward_lower_bound <
                        rewards) & (rewards < self.reward_upper_bound)
            if not torch.any(in_range):
                return Categorical(logits=torch.ones(self.task_size))
            in_range = in_range.float()
            return Categorical(probs=in_range / in_range.sum())

    def update(self, rewards, step=None):
        for task, reward in rewards.items():
            self.histories[task].append(reward)
