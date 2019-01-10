import torch
import torch.nn as nn
from gym.spaces import Box
from utils import space_to_size

from ppo.utils import init, init_normc_, mlp


class GAN:
    def __init__(self, hidden_size: int, learning_rate: float, entropy_coef:
    float, goal_space: Box,
                 **kwargs):
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        self.goal_space = goal_space
        self.hidden_size = hidden_size
        self.network = nn.Sequential(mlp(num_inputs=hidden_size,
                                         num_outputs=space_to_size(goal_space),
                                         hidden_size=hidden_size, **kwargs),
                                     torch.nn.Sigmoid())
        self.regularizer = None

    def sample(self, num_outputs):
        mean = torch.zeros(num_outputs, self.hidden_size)
        std = torch.ones(num_outputs, self.hidden_size)
        dist = torch.distributions.Normal(mean, std)
        noise = dist.sample()
        log_prob = dist.log_prob(noise).sum(dim=-1).exp()

        goal = self.network(noise)
        high, low = [torch.tensor(x).view(1, -1) for x in
                     [self.goal_space.high, self.goal_space.low]]
        goal = goal * (high - low) + low
        if self.regularizer is None:
            self.regularizer = log_prob.mean()
        else:
            self.regularizer += .01 * (log_prob.mean() - self.regularizer)
        importance_weighting = self.regularizer / log_prob
        return noise, goal, importance_weighting.view(-1, 1)

    def parameters(self):
        return self.network.parameters()

    def to(self, device):
        self.network.to(device)
