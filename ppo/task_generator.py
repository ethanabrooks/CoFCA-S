from enum import Enum

import torch

from ppo.util import Categorical, NoInput, init_normc_

SamplingStrategy = Enum('SamplingStrategy', 'baseline adaptive')


class TaskGenerator(NoInput):
    def __init__(self, task_size, temperature: float, exploration_bonus: float,
                 sampling_strategy):
        super().__init__(task_size)
        self.exploration_bonus = exploration_bonus
        self.sampling_strategy = sampling_strategy
        self.task_size = task_size
        self.temperature = temperature
        if sampling_strategy == SamplingStrategy.adaptive.name:
            self.logits = torch.Tensor(1, task_size)
            init_normc_(self.logits)
            self.logits = self.logits.view(-1)
        else:
            self.logits = torch.ones(task_size)

    def dist(self):
        return Categorical(logits=self.temperature * self.logits)

    def sample(self):
        return self.dist().sample()

    def probs(self):
        return self.dist().probs.detach().numpy()

    def importance_weight(self, probs):
        return 1 / (self.task_size * probs)

    def update(self, tasks, grads):
        if self.sampling_strategy == SamplingStrategy.adaptive.name:
            self.logits += self.exploration_bonus
            self.logits[tasks] = grads
