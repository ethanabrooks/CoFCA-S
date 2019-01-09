import torch
import torch.nn as nn
from gym.spaces import Box

from ppo.utils import mlp
from utils import space_to_size

from ppo.utils import mlp


class GAN(nn.Module):
    def __init__(self, goal_space: Box, hidden_size, learning_rate: float,
                 entropy_coef: float, **kwargs):
        super().__init__()
        self.learning_rate = learning_rate
        self.entropy_coef = entropy_coef
        self.goal_space = goal_space
        goal_size = space_to_size(goal_space)
        self.hidden_size = hidden_size
        self.network = nn.Sequential(
            mlp(num_inputs=hidden_size,
                hidden_size=hidden_size,
                num_outputs=2 * goal_size,
                name='gan',
                **kwargs))
        self.softplus = torch.nn.Softplus()
        self.regularizer = None

    def goal_input(self, num_outputs):
        return torch.ones((num_outputs, self.hidden_size))

    def forward(self, *inputs):
        raise NotImplementedError

    def dist(self, num_inputs):
        network_out = self.network(self.goal_input(num_inputs))
        a, b = torch.chunk(network_out, 2, dim=-1)
        return torch.distributions.Normal(a, self.softplus(b))

    def log_prob(self, goal):
        num_inputs = goal.size()[0]
        return self.dist(num_inputs).log_prob(goal)

    def sample(self, num_outputs):
        dist = self.dist(num_outputs)
        samples = dist.sample()
        high = torch.tensor(self.goal_space.high)
        low = torch.tensor(self.goal_space.low)
        goals = samples * (high - low) + low
        log_prob = dist.log_prob(samples).sum(dim=-1).exp()
        if self.regularizer is None:
            self.regularizer = log_prob.mean()
        else:
            self.regularizer += .01 * (log_prob.mean() - self.regularizer)
        importance_weighting = self.regularizer / log_prob
        return samples, goals, importance_weighting.view(-1, 1)

    def parameters(self, **kwargs):
        return self.network.parameters(**kwargs)

    def to(self, device):
        self.network.to(device)
