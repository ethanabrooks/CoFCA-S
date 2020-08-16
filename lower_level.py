import numpy as np
import torch
from gym import spaces
from torch import nn as nn
from torch.nn import functional as F

from layers import Flatten
import networks
from networks import NNBase
from lines import Subtask
from env import Obs, Env, subtasks
from utils import init_, init, init_normc_


def get_obs_sections(obs_spaces):
    return [int(np.prod(s.shape)) for s in obs_spaces]


class Agent(networks.Agent):
    def build_recurrent_module(self, hidden_size, network_args, obs_spaces, recurrent):
        return LowerLevel(
            obs_space=obs_spaces,
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
        obs_space,
        num_conv_layers,
        kernel_size,
        stride,
        activation=nn.ReLU(),
        **_,
    ):
        if type(obs_space) is spaces.Dict:
            obs_space = Obs(**obs_space.spaces)
        assert num_layers > 0
        H = hidden_size
        super().__init__(
            recurrent=recurrent, recurrent_input_size=H, hidden_size=hidden_size
        )
        self.register_buffer(
            "subtasks",
            torch.tensor(
                [Env.preprocess_line(Subtask(s)) for s in subtasks()] + [[0, 0, 0, 0]]
            ),
        )
        (d, h, w) = obs_space.obs.shape
        inventory_size = obs_space.inventory.n
        line_nvec = torch.tensor(obs_space.lines.nvec)
        offset = F.pad(line_nvec[0, :-1].cumsum(0), [1, 0])
        self.register_buffer("offset", offset)
        self.obs_spaces = obs_space
        self.obs_sections = get_obs_sections(self.obs_spaces)
        padding = (kernel_size // 2) % stride

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
            h = w = (h + (2 * padding) - (kernel_size - 1) - 1) // stride + 1
            kernel_size = min(h, kernel_size)
        self.conv.add_module(name="flatten", module=Flatten())
        init2 = lambda m: init(m, init_normc_, lambda x: nn.init.constant_(x, 0))

        self.conv_projection = nn.Sequential(
            init2(nn.Linear(h * w * hidden_size, hidden_size)), activation
        )
        self.line_embed = nn.EmbeddingBag(line_nvec[0].sum(), hidden_size)
        self.inventory_embed = nn.Sequential(
            init2(nn.Linear(inventory_size, hidden_size)), activation
        )

        self.mlp = nn.Sequential()
        in_size = hidden_size if recurrent else H
        for i in range(num_layers):
            self.mlp.add_module(
                name=f"fc{i}",
                module=nn.Sequential(
                    init2(nn.Linear(in_size, hidden_size)), activation
                ),
            )
            in_size = hidden_size

        self.critic_linear = init2(nn.Linear(in_size, 1))
        self._output_size = in_size
        self.train()

    def parse_inputs(self, inputs: torch.Tensor):
        return torch.split(inputs, self.obs_sections, dim=-1)

    @property
    def output_size(self):
        return self._output_size

    def forward(self, inputs, rnn_hxs, masks, upper=None):
        if not type(inputs) is Obs:
            inputs = Obs(*self.parse_inputs(inputs))
        N = inputs.obs.size(0)
        lines = inputs.lines.reshape(N, -1, self.obs_spaces.lines.shape[-1])
        if upper is None:
            R = torch.arange(N, device=inputs.obs.device)
            p = inputs.active.clamp(min=0, max=lines.size(1) - 1)
            line = lines[R, p.long().flatten()]
        else:
            # upper = torch.tensor([int((input("upper:")))])
            upper = torch.clamp(upper, 0, len(self.subtasks) - 1)
            line = self.subtasks[upper.long().flatten()]
        obs = inputs.obs.reshape(N, *self.obs_spaces.obs.shape)
        lines_embed = self.line_embed(line.long() + self.offset)
        obs_embed = self.conv_projection(self.conv(obs))
        inventory_embed = self.inventory_embed(inputs.inventory)
        x = lines_embed * obs_embed * inventory_embed

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden = self.mlp(x)

        return self.critic_linear(hidden), hidden, rnn_hxs
