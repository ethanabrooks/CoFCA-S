import numpy as np
import torch

from ppo.util import Categorical, NoInput


class TaskGenerator(NoInput):
    def __init__(self, task_size, learning_rate: float, entropy_coef: float,
                 temperature: float, **kwargs):
        super().__init__(task_size)
        self.learning_rate = learning_rate
        self.entropy_coef = entropy_coef
        self.task_size = task_size
        self.softmax = torch.nn.Softmax(dim=-1)
        self.temperature = temperature
        self.counter = np.ones(task_size)
        self.time_since_selected = np.ones(task_size)

    def sample(self, num_samples):
        self.time_since_selected += 1
        choices = np.random.choice(
            self.task_size,
            size=num_samples,
            replace=False,
            p=self.probs().detach().numpy())
        self.time_since_selected[choices] = 1
        self.counter[choices] += 1
        return choices

    def exploration_bonus(self):
        return torch.tensor(
            np.sqrt(np.log(self.time_since_selected) / self.counter),
            dtype=torch.float)

    def probs(self):
        return Categorical(
            logits=self.temperature * self.weight).probs.view(-1)

    def importance_weight(self, task_index):
        return 1 / (self.task_size * self.probs()[task_index]).detach()
