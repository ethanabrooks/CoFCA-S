from gym.spaces import Box, Discrete
import torch
import torch.nn as nn

from ppo.util import Categorical, NoInput, init_normc_, mlp
from utils import space_to_size


class TaskGenerator(nn.Module):
    def __init__(self, task_space: Box, hidden_size, learning_rate: float,
                 entropy_coef: float, num_samples: int, **kwargs):
        super().__init__()
        self.num_samples = num_samples
        self.learning_rate = learning_rate
        self.entropy_coef = entropy_coef
        self.task_space = task_space
        self.task_size = task_size = space_to_size(task_space)
        self.hidden_size = hidden_size
        input_size = task_size + 1
        if isinstance(self.task_space, Box):
            num_outputs = 2 * task_size
        else:
            num_outputs = task_size
        self.takes_input = bool(hidden_size)
        if not hidden_size:
            self.network = NoInput(num_outputs)
        else:
            self.network = nn.Sequential(
                mlp(num_inputs=input_size,
                    hidden_size=hidden_size,
                    num_outputs=num_outputs,
                    name='task_network',
                    **kwargs))
        self.learning_rate_regularizer = 1 / num_outputs
        self.softplus = torch.nn.Softplus()
        self.regularizer = None
        self.input = torch.rand(input_size)

    def set_input(self, task, norm):
        if isinstance(self.task_space, Discrete):
            task_vector = torch.zeros(self.task_size)
            task_vector[task.int().squeeze()] = 1
            task = task_vector
        self.input = torch.cat((task, norm.expand(1)))
        self.input = torch.ones_like(self.input)

    def forward(self, *inputs):
        raise NotImplementedError

    def dist(self, num_inputs):
        network_out = self.network(self.input.repeat(num_inputs, 1))
        if isinstance(self.task_space, Box):
            network_out = self.softplus(network_out)
            a, b = torch.chunk(network_out, 2, dim=-1)
            return torch.distributions.Beta(a, b)
        else:
            return Categorical(logits=network_out + 1)

    def sample(self, num_outputs):
        dist = self.dist(num_outputs)
        samples = dist.sample()
        if isinstance(self.task_space, Box):
            high = torch.tensor(self.task_space.high)
            low = torch.tensor(self.task_space.low)
            tasks = samples * (high - low) + low
            raise NotImplementedError
        else:
            prob = dist.log_prob(samples.squeeze(-1)).exp()
            importance_weighting = (self.learning_rate_regularizer / prob)
            tasks = samples
        return samples, tasks, importance_weighting

    def parameters(self, **kwargs):
        return self.network.parameters(**kwargs)

    def to(self, device):
        self.network.to(device)
