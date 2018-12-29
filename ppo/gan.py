import torch
import torch.nn as nn
from gym.spaces import Box

from ppo.utils import init, init_normc_


class GAN(nn.Module):
    def forward(self, *input):
        return self.network(*input)

    def __init__(self, num_layers, hidden_size, activation, goal_size, goal_space: Box):
        super().__init__()
        self.goal_space = goal_space
        self.hidden_size = hidden_size
        self.network = nn.Sequential()

        def linear(size):
            return init(
                nn.Linear(hidden_size, size), init_normc_,
                lambda x: nn.init.constant_(x, 0))

        for i in range(num_layers):
            if i < num_layers - 1:
                self.network.add_module(
                    name=f'linear{i}', module=linear(hidden_size))
                self.network.add_module(
                    name=f'activation{i}', module=activation)
            else:
                # last layer: no activation
                self.network.add_module(
                    name=f'linear{i}', module=linear(goal_size))

    def sample(self, num_outputs):
        mean = torch.zeros(num_outputs, self.hidden_size)
        std = torch.ones(num_outputs, self.hidden_size)
        noise = torch.normal(mean, std)
        params = self(noise)
        high = torch.from_numpy(self.goal_space.high)
        low = torch.from_numpy(self.goal_space.low)
        squashed = torch.sigmoid(params) * (high - low) + low
        assert self.goal_space.contains(squashed.squeeze().detach().numpy())
        return squashed

    def parameters(self):
        return self.network.parameters()

    def to(self, device):
        self.network.to(device)
