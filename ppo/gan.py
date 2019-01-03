import torch
import torch.nn as nn
from gym.spaces import Box
from utils import space_to_size

from ppo.utils import init, init_normc_
from ppo.distributions import DiagGaussian


class GAN(nn.Module):
    def __init__(self, num_layers, hidden_size, activation, goal_size,
                 goal_space: Box):
        super().__init__()
        self.goal_space = goal_space
        self.hidden_size = hidden_size
        self.network = nn.Sequential()
        self.dist = DiagGaussian(self.hidden_size, space_to_size(goal_space))

        def linear(size):
            return init(
                nn.Linear(hidden_size, size), init_normc_,
                lambda x: nn.init.constant_(x, 0))

        for i in range(num_layers):
            self.network.add_module(
                name=f'linear{i}', module=linear(hidden_size))
            self.network.add_module(
                name=f'activation{i}', module=activation)

    def goal_input(self, num_outputs):
        return torch.ones((num_outputs, self.hidden_size))

    def forward(self, *inputs):
        dist = self.dist(self.network(*inputs))
        goal = dist.sample()
        high = torch.from_numpy(self.goal_space.high)
        low = torch.from_numpy(self.goal_space.low)
        squashed = torch.sigmoid(goal) * (high - low) + low
        # assert self.goal_space.contains(squashed.squeeze().detach().numpy())
        return squashed, dist.log_probs(goal)

    def log_prob(self, goal):
        num_inputs = goal.size()[0]
        params = self.network(self.goal_input(num_inputs))
        dist = self.dist(params)
        return dist.log_probs(goal)

    def sample(self, num_outputs):
        return self(self.goal_input(num_outputs))

    def parameters(self):
        return self.network.parameters()

    def to(self, device):
        self.network.to(device)
