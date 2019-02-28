from gym.spaces import Box, Discrete
import torch
import torch.nn as nn

from ppo.util import Categorical, NoInput, init_normc_, mlp
from utils import space_to_size


class TaskGenerator(nn.Module):
    def __init__(self, task_space: Box, learning_rate: float,
                 entropy_coef: float, **kwargs):
        super().__init__()
        self.learning_rate = learning_rate
        self.entropy_coef = entropy_coef
        self.task_space = task_space
        self.task_size = task_size = space_to_size(task_space)
        input_size = task_size + 1
        if isinstance(self.task_space, Box):
            raise NotImplementedError

        self.softplus = torch.nn.Softplus()
        self.input = torch.rand(input_size)
        self.network = NoInput(task_size)

    def forward(self, *inputs):
        raise NotImplementedError

    def dist(self, num_inputs):
        if isinstance(self.task_space, Box):
            network_out = self.softplus(network_out)
            a, b = torch.chunk(network_out, 2, dim=-1)
            raise NotImplementedError
            # return torch.distributions.Beta(a, b)
        else:
            return Categorical(
                logits=self.network.parameter.repeat(num_inputs, 1))

    def sample(self, num_outputs):
        dist = self.dist(num_outputs)
        tasks = dist.sample().view(num_outputs)
        if isinstance(self.task_space, Box):
            high = torch.tensor(self.task_space.high)
            low = torch.tensor(self.task_space.low)
            raise NotImplementedError
        else:
            prob = dist.log_prob(tasks).exp()
            importance_weighting = 1 / (self.task_size * prob)
        return tasks, importance_weighting

    def parameters(self, **kwargs):
        return self.network.parameters(**kwargs)

    def to(self, device):
        self.network.to(device)
