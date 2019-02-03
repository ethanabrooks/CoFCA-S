import torch
import torch.nn as nn
from gym.spaces import Box

from ppo.util import mlp
from utils import space_to_size

from ppo.util import mlp


class GAN(nn.Module):
    def __init__(self, goal_space: Box, hidden_size, learning_rate: float,
                 entropy_coef: float, **kwargs):
        super().__init__()
        self.learning_rate = learning_rate
        self.entropy_coef = entropy_coef
        self.goal_space = goal_space
        goal_size = space_to_size(goal_space)
        self.hidden_size = hidden_size
        input_size = goal_size + 1
        self.network = nn.Sequential(
            mlp(num_inputs=input_size,
                hidden_size=hidden_size,
                num_outputs=2 * goal_size,
                name='gan',
                **kwargs))
        self.softplus = torch.nn.Softplus()
        self.regularizer = None
        self.input = torch.rand(input_size)

    def set_input(self, goal, norm):
        self.input = torch.cat((goal, norm.expand(1)))

    def forward(self, *inputs):
        raise NotImplementedError

    def dist(self, num_inputs):
        network_out = self.softplus(self.network(self.input.repeat(num_inputs, 1)))
        a, b = torch.chunk(network_out, 2, dim=-1)
        return torch.distributions.Beta(a, b)

    def sample(self, num_outputs):
        dist = self.dist(num_outputs)
        samples = dist.sample()
        high = torch.tensor(self.goal_space.high)
        low = torch.tensor(self.goal_space.low)
        goals = samples * (high - low) + low
        prob = dist.log_prob(samples).sum(-1).exp()
        importance_weighting = 1 / prob
        return samples, goals, importance_weighting.view(-1, 1)

    def parameters(self, **kwargs):
        return self.network.parameters(**kwargs)

    def to(self, device):
        self.network.to(device)
