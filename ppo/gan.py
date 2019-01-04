import torch
import torch.nn as nn
from gym.spaces import Box
from utils import space_to_size

from ppo.utils import mlp


class GAN(nn.Module):
    def __init__(self, goal_space: Box, hidden_size, **kwargs):
        super().__init__()
        self.goal_space = goal_space
        goal_size = space_to_size(goal_space)
        self.hidden_size = hidden_size
        self.network = nn.Sequential(mlp(num_inputs=hidden_size,
                                         hidden_size=hidden_size,
                                         num_outputs=2 * goal_size,
                                         name='gan', **kwargs),
                                     torch.nn.Softplus())
        self.uniform_dist = torch.distributions.Uniform(low=torch.tensor(goal_space.low),
                                                        high=torch.tensor(
                                                            goal_space.high))

    def goal_input(self, num_outputs):
        return torch.ones((num_outputs, self.hidden_size))

    def forward(self, *inputs):
        raise NotImplementedError

    def log_prob(self, goal):
        num_inputs = goal.size()[0]
        params = self.network(self.goal_input(num_inputs))
        dist = self.dist(params)
        return dist.log_probs(goal)

    def sample(self, num_outputs):
        network_out = self.network(self.goal_input(num_outputs))
        params = torch.chunk(network_out, 2, dim=-1)
        dist = torch.distributions.Beta(*params)
        sample = dist.sample()
        squashed = sample.numpy() * (
                self.goal_space.high - self.goal_space.low) + self.goal_space.low
        return squashed, dist.log_prob(sample)

    def parameters(self):
        return self.network.parameters()

    def to(self, device):
        self.network.to(device)
