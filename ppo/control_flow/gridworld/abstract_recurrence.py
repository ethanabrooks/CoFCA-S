import numpy as np
import torch
import torch.nn.functional as F
from gym import spaces
from torch import nn as nn

from ppo.utils import init_


class Recurrence(nn.Module):
    def __init__(self, conv_hidden_size, use_conv):
        super().__init__()
        self.conv_hidden_size = conv_hidden_size
        self.use_conv = use_conv
        d = self.obs_spaces.obs.shape[0]
        if use_conv:
            # layers = [
            #     nn.Conv2d(
            #         d,
            #         conv_hidden_size,
            #         kernel_size=kernel_size,
            #         stride=2 if kernel_size == 2 else 1,
            #         padding=0,
            #     ),
            #     nn.ReLU(),
            # ]
            # if kernel_size < 4:
            #     layers += [
            #         nn.Conv2d(
            #             conv_hidden_size,
            #             conv_hidden_size,
            #             kernel_size=2,
            #             stride=2,
            #             padding=0,
            #         ),
            #         nn.ReLU(),
            #     ]
            self.conv = nn.Sequential(
                nn.Conv2d(d, conv_hidden_size, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(conv_hidden_size, conv_hidden_size, kernel_size=4),
                nn.ReLU(),
            )
        else:
            self.conv = nn.Sequential(init_(nn.Linear(d, conv_hidden_size)), nn.ReLU())
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

    def build_memory(self, N, T, inputs):
        lines = inputs.lines.view(T, N, *self.obs_spaces.lines.shape)
        lines = lines.long()[0, :, :] + self.offset
        return self.embed_task(lines.view(-1, self.obs_spaces.lines.nvec[0].size)).view(
            *lines.shape[:2], self.encoder_hidden_size
        )  # n_batch, n_lines, hidden_size

    def preprocess_obs(self, obs):
        N = obs.size(0)
        if self.use_conv:
            return self.conv(obs).view(N, -1)
        else:
            return (
                self.conv(obs.permute(0, 2, 3, 1))
                .view(N, -1, self.conv_hidden_size)
                .max(dim=1)
                .values
            )
