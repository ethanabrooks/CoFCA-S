import numpy as np
import torch
import torch.nn.functional as F
from gym import spaces
from torch import nn as nn

import ppo.control_flow.recurrence
from ppo.control_flow.recurrence import RecurrentState


class Recurrence(ppo.control_flow.recurrence.Recurrence):
    def build_embed_task(self, hidden_size):
        return nn.Embedding(self.obs_spaces.lines.nvec[0], hidden_size)

    @property
    def gru_in_size(self):
        return 1 + 2 * self.task_embed_size + self.hidden_size

    @staticmethod
    def eval_lines_space(n_eval_lines, train_lines_space):
        return spaces.MultiDiscrete(
            np.repeat(train_lines_space.nvec[:1], repeats=n_eval_lines, axis=0)
        )

    def pack(self, hxs):
        def pack():
            for name, size, hx in zip(
                RecurrentState._fields, self.state_sizes, zip(*hxs)
            ):
                x = torch.stack(hx).float()
                assert np.prod(x.shape[2:]) == size
                yield x.view(*x.shape[:2], -1)

        hx = torch.cat(list(pack()), dim=-1)
        return hx, hx[-1:]

    def parse_hidden(self, hx: torch.Tensor) -> RecurrentState:
        return RecurrentState(*torch.split(hx, self.state_sizes, dim=-1))

    def inner_loop(self, inputs, rnn_hxs):
        T, N, dim = inputs.shape
        inputs, actions = torch.split(
            inputs.detach(), [dim - self.action_size, self.action_size], dim=2
        )

        # parse non-action inputs
        inputs = self.parse_obs(inputs)
        inputs = inputs._replace(obs=inputs.obs.view(T, N, *self.obs_spaces.obs.shape))

        M = self.build_memory(N, T, inputs)
        G, H = self.task_encoder(M)
        H = H.transpose(0, 1).reshape(N, -1)

        new_episode = torch.all(rnn_hxs == 0, dim=-1).squeeze(0)
        hx = self.parse_hidden(rnn_hxs)
        for _x in hx:
            _x.squeeze_(0)

        h = hx.h
        hx.a[new_episode] = self.n_a - 1
        A = torch.cat([actions[:, :, 0], hx.a.view(1, N)], dim=0).long()

        for t in range(T):
            obs = self.preprocess_obs(inputs.obs[t])
            x = [obs, H, self.embed_upper(A[t - 1].clone())]
            h = self.gru(torch.cat(x, dim=-1), h)
            z = F.relu(self.zeta2(h))
            a_dist = self.actor(z)
            self.sample_new(A[t], a_dist)

            yield RecurrentState(
                a=A[t],
                v=self.critic(z),
                h=h,
                p=hx.p,
                a_probs=a_dist.probs,
                d=hx.d,
                d_probs=hx.d_probs,
            )
