from collections import namedtuple

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

import ppo.events as events
import ppo.oh_et_al
from ppo.distributions import Categorical, FixedCategorical
from ppo.layers import Parallel, Reshape, Product, Flatten
from ppo.utils import init_

RecurrentState = namedtuple("RecurrentState", "a a_probs v h p")


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
        use_M_plus_minus,
        feed_r_initially,
        oh_et_al,
    ):
        super().__init__()
        self.oh_et_al = oh_et_al
        self.feed_r_initially = feed_r_initially = feed_r_initially or baseline
        self.use_M_plus_minus = use_M_plus_minus and (not baseline)
        self.baseline = baseline
        obs_spaces = (ppo.oh_et_al.gridworld.Obs if oh_et_al else events.wrapper.Obs)(
            **observation_space.spaces
        )
        self.obs_shape = d, h, w = obs_spaces.base.shape
        self.obs_sections = [int(np.prod(s.shape)) for s in obs_spaces]
        self.action_size = 1
        self.debug = debug

        # networks
        if oh_et_al:
            self.task_embeddings = nn.EmbeddingBag(
                obs_spaces.subtasks.nvec[0].sum(), hidden_size
            )
            self.subtasks_nvec = obs_spaces.subtasks.nvec
        else:
            self.task_embeddings = nn.Embedding(
                obs_spaces.instructions.nvec[0], hidden_size
            )
            self.subtasks_nvec = None
        self.task_output_sections = [hidden_size]
        if not baseline:
            self.task_output_sections += [1, 1]
        if self.use_M_plus_minus:
            self.task_output_sections += [hidden_size, hidden_size]

        self.task_encoder = nn.GRU(
            int(hidden_size), int(sum(self.task_output_sections))
        )
        self.f = nn.Sequential(
            *(
                [
                    Parallel(Reshape(1, d, h, w), Reshape(hidden_size, 1, 1, 1)),
                    Product(),
                    Reshape(d * hidden_size, h, w),
                    init_(
                        nn.Conv2d(d * hidden_size, hidden_size, kernel_size=1),
                        activation,
                    ),
                ]
                if feed_r_initially
                else [init_(nn.Conv2d(d, hidden_size, kernel_size=1), activation)]
            ),
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
            *(
                []
                if feed_r_initially
                else [init_(nn.Linear(hidden_size * h * w, hidden_size))]
            ),
            activation,
        )
        self.gru = nn.GRUCell(
            hidden_size * h * w if feed_r_initially else hidden_size, hidden_size
        )
        if not self.baseline:
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
            h=hidden_size,
            p=len(obs_spaces.subtasks.nvec)
            if oh_et_al
            else obs_spaces.instructions.nvec.size,
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

    def parse_inputs(self, inputs: torch.Tensor):
        return (ppo.oh_et_al.gridworld.Obs if self.oh_et_al else events.wrapper.Obs)(
            *torch.split(inputs, self.obs_sections, dim=-1)
        )

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
        if self.oh_et_al:
            n_subtasks = len(self.subtasks_nvec)
            subtask_size = self.subtasks_nvec[0].size
            task = inputs.subtasks[0].reshape(N * n_subtasks, subtask_size).long()
            embeddings = self.task_embeddings(task).view(N, n_subtasks, -1)
        else:
            embeddings = self.task_embeddings(inputs.instructions[0].long())
        rnn_inputs = embeddings.transpose(0, 1)
        X, _ = self.task_encoder(rnn_inputs)

        new_episode = torch.all(rnn_hxs == 0, dim=-1).squeeze(0)
        hx = self.parse_hidden(rnn_hxs)
        for _x in hx:
            _x.squeeze_(0)

        encoding = X.transpose(0, 1).split(self.task_output_sections, dim=-1)
        M = encoding[0]
        h = hx.h
        p = hx.p
        if not self.baseline:
            p0 = encoding[1].squeeze(-1)
            c = encoding[2].squeeze(-1)
            p[new_episode] = p0[new_episode]

        M_minus = encoding[3] if self.use_M_plus_minus else M
        M_plus = encoding[4] if self.use_M_plus_minus else M
        A = torch.cat([actions, hx.a.unsqueeze(0)], dim=0).long().squeeze(2)

        for t in range(T):
            if self.baseline:
                r = M.sum(1)
            else:
                p = F.softmax(p, dim=-1)
                r = (p.unsqueeze(1) @ M).squeeze(1)
            if self.feed_r_initially:
                # noinspection PyTypeChecker
                h = self.gru(self.f((inputs.base[t], r)), h)
            else:
                x = self.f(inputs.base[t])
                h = self.gru(r * x, h)

            dist = self.actor(h)
            if not self.oh_et_al:
                nonzero = F.pad(inputs.interactable[t], [5, 0], "constant", 1)
                probs = dist.probs * nonzero

                # noinspection PyTypeChecker
                deficit = 1 - probs.sum(-1, keepdim=True)
                probs = probs + F.normalize(nonzero, p=1, dim=-1) * deficit
                dist = FixedCategorical(
                    probs=F.normalize(torch.clamp(probs, 0.0, 1.0), p=1, dim=-1)
                )

            self.sample_new(A[t], dist)
            if not self.baseline:
                a = self.a_one_hots(A[t].flatten().long())
                e = self.psi((h, a)).unsqueeze(1).expand(*M.shape)
                self.print("c", c)
                self.print("p1", p)
                p_plus = F.cosine_similarity(e, M_plus, dim=-1)
                p_minus = F.cosine_similarity(e, M_minus, dim=-1)
                self.print("minus")
                self.print(p_minus)
                self.print("plus")
                self.print(p_plus)
                p = p + c * (p_plus - p_minus)
                self.print("p2", p)
            yield RecurrentState(a=A[t], a_probs=dist.probs, v=self.critic(h), h=h, p=p)
