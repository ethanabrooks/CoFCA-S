import torch
import torch.nn as nn
from gym.spaces import Box
from utils import space_to_size

from ppo.utils import init, init_normc_, mlp


class GAN(nn.Module):
    def forward(self, *input):
        self.network(*input)

    def __init__(self, learning_rate: float, entropy_coef:
    float, goal_space: Box, **kwargs):
        super().__init__()
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        self.goal_space = goal_space
        self.goal_size = space_to_size(goal_space)
        self.network = nn.Sequential(mlp(num_inputs=self.goal_size,
                                         num_outputs=self.goal_size,
                                         **kwargs),
                                     torch.nn.Sigmoid())
        self.regularizer = None

    def sample(self, num_outputs):
        mean = torch.zeros(num_outputs, self.goal_size)
        std = torch.ones(num_outputs, self.goal_size)
        dist = torch.distributions.Normal(mean, std)
        noise = dist.sample()
        prob = dist.log_prob(noise).sum(dim=-1).exp()

        goal = self.network(noise)
        high, low = [torch.tensor(x).view(1, -1) for x in
                     [self.goal_space.high, self.goal_space.low]]
        goal = goal * (high - low) + low
        if self.regularizer is None:
            self.regularizer = prob.mean()
        else:
            self.regularizer += .01 * (prob.mean() - self.regularizer)
        eps = torch.tensor(1e-5)
        importance_weighting = self.regularizer / torch.max(prob, eps)
        return goal, importance_weighting.view(-1, 1)

    def parameters(self, **kwargs):
        return self.network.parameters(**kwargs)

    def to(self, device):
        self.network.to(device)
