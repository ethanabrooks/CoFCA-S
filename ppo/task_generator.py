from gym.spaces import Box, Discrete
import torch.nn as nn

from ppo.util import Categorical, NoInput, init_normc_, mlp
from utils import space_to_size


class TaskGenerator(nn.Module):
    def __init__(self, task_size, learning_rate: float, entropy_coef: float,
                 **kwargs):
        super().__init__()
        self.learning_rate = learning_rate
        self.entropy_coef = entropy_coef
        self.task_size = task_size
        self.network = NoInput(task_size)

    def forward(self, *inputs):
        raise NotImplementedError

    def sample(self, num_outputs):
        dist = Categorical(
            logits=self.network.parameter.repeat(num_outputs, 1))
        tasks = dist.sample().view(num_outputs)
        prob = dist.log_prob(tasks).exp()
        importance_weighting = 1 / (self.task_size * prob)
        return tasks, importance_weighting

    def parameters(self, **kwargs):
        return self.network.parameters(**kwargs)

    def to(self, device):
        self.network.to(device)
