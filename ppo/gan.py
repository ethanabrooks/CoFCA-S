import torch
import torch.nn as nn
from gym.spaces import Box, Discrete
from utils import space_to_size

from ppo.util import mlp


class GAN(nn.Module):
    def __init__(self, goal_space: Box, hidden_size, learning_rate: float,
                 entropy_coef: float, **kwargs):
        super().__init__()
        self.learning_rate = learning_rate
        self.entropy_coef = entropy_coef
        self.goal_space = goal_space
        self.goal_size = goal_size = space_to_size(goal_space)
        self.hidden_size = hidden_size
        input_size = goal_size + 1
        if isinstance(self.goal_space, Box):
            num_outputs = 2 * goal_size
        else:
            num_outputs = goal_size
        self.network = nn.Sequential(
            mlp(num_inputs=input_size,
                hidden_size=hidden_size,
                num_outputs=num_outputs,
                name='gan',
                **kwargs))
        self.softplus = torch.nn.Softplus()
        self.regularizer = None
        self.input = torch.rand(input_size)

    def set_input(self, goal, norm):
        if isinstance(self.goal_space, Discrete):
            goal_vector = torch.zeros(self.goal_size)
            goal_vector[goal.int().squeeze()] = 1
            goal = goal_vector
        self.input = torch.cat((goal, norm.expand(1)))

    def forward(self, *inputs):
        raise NotImplementedError

    def dist(self, num_inputs):
        network_out = self.softplus(self.network(self.input.repeat(num_inputs, 1)))
        if isinstance(self.goal_space, Box):
            a, b = torch.chunk(network_out, 2, dim=-1)
            return torch.distributions.Beta(a, b)
        else:
            return torch.distributions.Categorical(logits=network_out)

    def sample(self, num_outputs):
        # dist = self.dist(num_outputs)
        samples = torch.tensor([self.goal_space.sample()
                                for _ in range(num_outputs)])  #
        # dist.sample()
        prob = torch.ones(num_outputs)  # dist.log_prob(samples).sum(
        # -1).exp()
        importance_weighting = (1 / prob).view(-1, 1).squeeze(-1)
        if isinstance(self.goal_space, Box):
            high = torch.tensor(self.goal_space.high)
            low = torch.tensor(self.goal_space.low)
            goals = samples * (high - low) + low
        else:
            goals = samples
        return samples, goals, importance_weighting

    def parameters(self, **kwargs):
        return self.network.parameters(**kwargs)

    def to(self, device):
        self.network.to(device)
