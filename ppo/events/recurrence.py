from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn as nn
from torch.nn import functional as F

from ppo.distributions import Categorical, FixedCategorical
from ppo.events.wrapper import Obs
from ppo.layers import Parallel, Reshape, Product, Flatten
from ppo.utils import init_

RecurrentState = namedtuple("RecurrentState", "a a_probs v s p")


class Recurrence(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        activation,
        hidden_size,
        num_layers,
        debug,
        baseline,
    ):
        super().__init__()
        self.baseline = baseline
        obs_spaces = Obs(**observation_space.spaces)
        self.obs_shape = d, h, w = obs_spaces.base.shape
        self.obs_sections = [int(np.prod(s.shape)) for s in obs_spaces]
        self.action_size = 1
        self.debug = debug

        # networks
        self.task_embeddings = nn.Embedding(obs_spaces.subtasks.nvec[0], hidden_size)
        self.parser_sections = [1, 1] + [hidden_size] * 3
        self.parser = nn.GRU(int(hidden_size), int(sum(self.parser_sections)))
        self.f = nn.Sequential(
            Parallel(Reshape(1, d, h, w), Reshape(hidden_size, 1, 1, 1)),
            Product(),
            Reshape(d * hidden_size, h, w),
            init_(nn.Conv2d(d * hidden_size, hidden_size, kernel_size=1), activation),
            activation,
            *[
                nn.Sequential(
                    init_(
                        nn.Conv2d(hidden_size, hidden_size, kernel_size=1), activation
                    ),
                    activation,
                )
                for _ in range(num_layers)
            ],
            activation,
            Flatten(),
            init_(nn.Linear(hidden_size * h * w, hidden_size)),
            activation,
        )
        self.psi = nn.Sequential(
            Parallel(Reshape(hidden_size, 1), Reshape(1, action_space.n)),
            Product(),
            Reshape(hidden_size * action_space.n),
            init_(nn.Linear(hidden_size * action_space.n, hidden_size)),
        )
        self.critic = init_(nn.Linear(hidden_size, 1))
        self.actor = Categorical(hidden_size, action_space.n)
        self.a_one_hots = nn.Embedding.from_pretrained(torch.eye(action_space.n))

        self.state_sizes = RecurrentState(
            a=1,
            a_probs=action_space.n,
            v=1,
            s=hidden_size,
            p=obs_spaces.subtasks.nvec.size,
        )

    @staticmethod
    def sample_new(x, dist):
        new = x < 0
        x[new] = dist.sample()[new].flatten()

    def forward(self, inputs, hx):
        return self.pack(self.inner_loop(inputs, rnn_hxs=hx))

    @staticmethod
    def pack(hxs):
        def pack():
            for name, hx in RecurrentState(*zip(*hxs))._asdict().items():
                x = torch.stack(hx).float()
                yield x.view(*x.shape[:2], -1)

        hx = torch.cat(list(pack()), dim=-1)
        return hx, hx[-1:]

    def parse_inputs(self, inputs: torch.Tensor) -> Obs:
        return Obs(*torch.split(inputs, self.obs_sections, dim=-1))

    def parse_hidden(self, hx: torch.Tensor) -> RecurrentState:
        return RecurrentState(*torch.split(hx, self.state_sizes, dim=-1))

    def print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def inner_loop(self, inputs, rnn_hxs):
        T, N, D = inputs.shape
        inputs, actions = torch.split(
            inputs.detach(), [D - self.action_size, self.action_size], dim=2
        )

        # parse non-action inputs
        inputs = self.parse_inputs(inputs)
        inputs = inputs._replace(base=inputs.base.view(T, N, *self.obs_shape))

        # build memory
        rnn_inputs = self.task_embeddings(inputs.subtasks[0].long()).transpose(0, 1)
        X, _ = self.parser(rnn_inputs)
        c, p0, M, M_minus, M_plus = X.transpose(0, 1).split(
            self.parser_sections, dim=-1
        )
        c.squeeze_(-1)
        p0.squeeze_(-1)

        new_episode = torch.all(rnn_hxs == 0, dim=-1).squeeze(0)
        hx = self.parse_hidden(rnn_hxs)
        for x in hx:
            x.squeeze_(0)
        p = hx.p
        p[new_episode] = p0[new_episode]
        A = torch.cat([actions, hx.a.unsqueeze(0)], dim=0).long().squeeze(2)
        for t in range(T):
            p = F.softmax(p, dim=-1)
            if self.baseline:
                r = M.sum(1)
            else:
                r = (p.unsqueeze(1) @ M).squeeze(1)
            s = self.f((inputs.base[t], r))
            dist = self.actor(s)
            nonzero = F.pad(inputs.interactable[t], [5, 0], "constant", 1)
            probs = dist.probs * nonzero
            deficit = 1 - probs.sum(-1, keepdim=True)
            probs = probs + F.normalize(nonzero, p=1, dim=-1) * deficit
            dist = FixedCategorical(
                probs=F.normalize(torch.clamp(probs, 0.0, 1.0), p=1, dim=-1)
            )
            self.sample_new(A[t], dist)
            a = self.a_one_hots(A[t].flatten().long())
            e = self.psi((s, a)).unsqueeze(1).expand(*M.shape)
            self.print("c", c)
            self.print("p1", p)
            p = p + c * F.cosine_similarity(e, M_plus, dim=-1)
            self.print("minus")
            self.print(-F.cosine_similarity(e, M_plus, dim=-1))
            self.print("p2", p)
            self.print("plus")
            self.print(F.cosine_similarity(e, M_minus, dim=-1))
            p = p - c * F.cosine_similarity(e, M_minus, dim=-1)
            self.print("p3", p)
            yield RecurrentState(a=A[t], a_probs=dist.probs, v=self.critic(s), s=s, p=p)
