import numpy as np
import torch
import torch.nn.functional as F
from gym import spaces
from torch import nn as nn

from ppo.utils import init_


class Recurrence:
    def __init__(
        self,
        conv_hidden_size,
        encoder_hidden_size,
        num_conv_layers,
        kernel_size,
        stride,
    ):
        self.conv_hidden_size = conv_hidden_size
        d, h, _ = self.obs_spaces.obs.shape

        def generate_convolutions(h):
            in_size = d + encoder_hidden_size + self.obs_spaces.inventory.n
            for _ in range(num_conv_layers):
                kernel = min(h, kernel_size)
                padding = (kernel // 2) % stride
                conv = init_(
                    nn.Conv2d(
                        in_channels=in_size,
                        out_channels=conv_hidden_size,
                        kernel_size=kernel,
                        stride=stride,
                        padding=padding,
                    )
                )
                h = int((h + (2 * padding) - (kernel - 1) - 1) // stride + 1)
                in_size = conv_hidden_size
                yield conv
                yield nn.ReLU()

        self.conv = nn.Sequential(*generate_convolutions(h))
        ones = torch.ones(1, dtype=torch.long)
        self.register_buffer("ones", ones)
        line_nvec = torch.tensor(self.obs_spaces.lines.nvec[0, :-1])
        offset = F.pad(line_nvec.cumsum(0), [1, 0])
        self.register_buffer("offset", offset)

    @property
    def gru_in_size(self):
        return self.hidden_size + self.conv_hidden_size + self.encoder_hidden_size

    @staticmethod
    def eval_lines_space(n_eval_lines, train_lines_space):
        return spaces.MultiDiscrete(
            np.repeat(train_lines_space.nvec[:1], repeats=n_eval_lines, axis=0)
        )

    def build_embed_task(self, hidden_size):
        return nn.EmbeddingBag(self.obs_spaces.lines.nvec[0].sum(), hidden_size)

    def preprocess_embed(self, N, T, inputs):
        lines = inputs.lines.view(T, N, *self.obs_spaces.lines.shape)
        lines = lines.long()[0, :, :] + self.offset
        return lines.view(-1, self.obs_spaces.lines.nvec[0].size)

    def preprocess_obs(self, obs):
        N = obs.size(0)
        return (
            self.conv(obs.permute(0, 2, 3, 1))
            .view(N, -1, self.conv_hidden_size)
            .max(dim=1)
            .values
        )
