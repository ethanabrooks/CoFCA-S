import numpy as np
import torch
from gym import spaces
from torch import nn as nn

import networks
from layers import Flatten
from lower_env import Obs
from networks import NNBase
from utils import init_, init, init_normc_


def optimal_padding(h, kernel, stride):
    n = np.ceil((h - kernel) / stride + 1)
    return 1 + int(np.ceil((stride * (n - 1) + kernel - h) / 2))


def conv_output_dimension(h, padding, kernel, stride, dilation=1):
    return int(1 + (h + 2 * padding - dilation * (kernel - 1) - 1) / stride)


def get_obs_sections(obs_spaces):
    return [int(np.prod(s.shape)) for s in obs_spaces]


class Agent(networks.Agent):
    def build_recurrent_module(self, hidden_size, network_args, obs_spaces, recurrent):
        return LowerLevel(
            obs_spaces=obs_spaces,
            recurrent=recurrent,
            hidden_size=hidden_size,
            **network_args,
        )


class LowerLevel(NNBase):
    def __init__(
        self,
        hidden_size,
        num_layers,
        recurrent,
        obs_spaces,
        num_conv_layers,
        kernel_size,
        stride,
        activation=nn.ReLU(),
    ):
        if type(obs_spaces) is spaces.Dict:
            obs_spaces = Obs(**obs_spaces.spaces)
        assert num_layers > 0
        super().__init__(
            recurrent=recurrent,
            recurrent_input_size=hidden_size,
            hidden_size=hidden_size,
        )
        self.obs_spaces = obs_spaces
        self.obs_sections = get_obs_sections(self.obs_spaces)

        (d, h, w) = obs_spaces.obs.shape
        assert h == w
        self.kernel_size = min(d, kernel_size)
        padding = optimal_padding(h, kernel_size, stride)

        self.conv = nn.Sequential()
        in_size = d
        assert num_conv_layers > 0
        for i in range(num_conv_layers):
            self.conv.add_module(
                name=f"conv{i}",
                module=nn.Sequential(
                    init_(
                        nn.Conv2d(
                            in_size,
                            hidden_size,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                        )
                    ),
                    activation,
                ),
            )
            in_size = hidden_size
            # h = w = (h + (2 * padding) - (kernel_size - 1) - 1) // stride + 1
            h = w = conv_output_dimension(h, padding, kernel_size, stride)
            kernel_size = min(h, kernel_size)
        self.conv.add_module(name="flatten", module=Flatten())
        _init = lambda m: init(m, init_normc_, lambda x: nn.init.constant_(x, 0))

        self.conv_projection = nn.Sequential(
            _init(nn.Linear(h * w * hidden_size, hidden_size)), activation
        )
        self.line_embed = networks.MultiEmbeddingBag(
            obs_spaces.line.nvec, embedding_dim=hidden_size
        )
        self.inventory_embed = networks.MultiEmbeddingBag(
            obs_spaces.inventory.nvec, embedding_dim=hidden_size
        )

        self.mlp = nn.Sequential()
        in_size = hidden_size if recurrent else hidden_size
        for i in range(num_layers):
            self.mlp.add_module(
                name=f"fc{i}",
                module=nn.Sequential(
                    _init(nn.Linear(in_size, hidden_size)), activation
                ),
            )
            in_size = hidden_size

        self.critic_linear = _init(nn.Linear(in_size, 1))
        self._output_size = in_size
        self.train()

    def parse_inputs(self, inputs: torch.Tensor):
        return torch.split(inputs, self.obs_sections, dim=-1)

    @property
    def output_size(self):
        return self._output_size

    def forward(self, inputs, rnn_hxs, masks):
        if not type(inputs) is Obs:
            inputs = Obs(*self.parse_inputs(inputs))
        N = inputs.obs.size(0)
        obs = inputs.obs.reshape(N, *self.obs_spaces.obs.shape)
        lines_embed = self.line_embed(inputs.line.long())
        obs_embed = self.conv_projection(self.conv(obs))
        inventory_embed = self.inventory_embed(inputs.inventory.long())
        x = lines_embed + obs_embed + inventory_embed

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden = self.mlp(x)

        return self.critic_linear(hidden), hidden, rnn_hxs
