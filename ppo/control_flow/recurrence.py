import functools
import itertools
from collections import namedtuple

import numpy as np
import torch
import torch.jit
from gym.spaces import Box, Discrete
from torch import nn as nn
from torch.nn import functional as F

import ppo
from gridworld_env.control_flow_gridworld import LineTypes
from gridworld_env.control_flow_gridworld import Obs
from ppo.control_flow.lower_level import (
    LowerLevel,
    g_binary_to_discrete,
    g_discrete_to_binary,
)
from ppo.control_flow.wrappers import Actions
from ppo.distributions import Categorical, DiagGaussian, FixedCategorical
from ppo.layers import (
    Concat,
    Flatten,
    Parallel,
    Product,
    Reshape,
    ShallowCopy,
    Sum,
    Times,
)
from ppo.utils import broadcast3d, init_, interp, trace, round

RecurrentState = namedtuple(
    "RecurrentState",
    "a g cr cg z l a_probs g_probs cr_probs cg_probs l_probs z_probs p r last_condition last_eval v",
)


class Recurrence(nn.Module):
    def __init__(
        self,
        obs_spaces,
        action_spaces,
        hidden_size,
        num_layers,
        recurrent,
        agent,
        debug,
        activation,
    ):
        super().__init__()
        self.debug = debug
        if agent:
            assert isinstance(agent, LowerLevel)
        self.agent = agent
        self.recurrent = recurrent
        self.obs_spaces = obs_spaces
        self.n_subtasks = self.obs_spaces.subtasks.nvec.shape[0]
        self.subtask_nvec = self.obs_spaces.subtasks.nvec[0]
        d, h, w = self.obs_shape = obs_spaces.base.shape
        self.obs_sections = [int(np.prod(s.shape)) for s in self.obs_spaces]
        self.line_size = int(self.subtask_nvec.sum())
        self.agent_subtask_size = int(self.subtask_nvec[:-2].sum())
        self.size_actions = [
            1 if isinstance(s, Discrete) else s.nvec.size for s in action_spaces
        ]
        # networks
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.conv1 = nn.Sequential(
            ShallowCopy(2),
            Parallel(
                Reshape(d, h * w),
                nn.Sequential(
                    init_(nn.Conv2d(d, 1, kernel_size=1)),
                    Reshape(1, h * w),  # TODO
                    nn.Softmax(dim=-1),
                ),
            ),
            Product(),
            Sum(dim=-1),
        )

        self.conv2 = nn.Sequential(
            Concat(dim=1),
            init_(nn.Conv2d(d + self.line_size, hidden_size, kernel_size=1), nn.ReLU()),
            nn.ReLU(),
            Flatten(),
        )
        self.xi = nn.Sequential(
            Parallel(
                nn.Sequential(Reshape(1, d, h, w)),
                nn.Sequential(Reshape(self.line_size, 1, 1, 1)),
            ),
            Product(),
            Reshape(self.line_size * d, h, w),
            nn.Sequential(
                init_(
                    nn.Conv2d(self.line_size * d, hidden_size, kernel_size=1),
                    activation,
                ),
                activation,
            ),
            *[
                nn.Sequential(
                    init_(
                        nn.Conv2d(hidden_size, hidden_size, kernel_size=1), activation
                    ),
                    activation,
                )
                for _ in range(num_layers - 1)
            ],
            # init_(nn.Conv2d(d, 32, 8, stride=4)), nn.ReLU(),
            # init_(nn.Conv2d(32, 64, kernel_size=4, stride=2)), nn.ReLU(),
            # init_(nn.Conv2d(32, 64, kernel_size=4, stride=2)), nn.ReLU(),
            # init_(nn.Conv2d(64, 32, kernel_size=3, stride=1)),
            Flatten(),
            # init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())
            init_(nn.Linear(hidden_size * h * w, hidden_size), nn.ReLU()),
            nn.ReLU(),
        )
        self.dist = nn.Sequential(
            init_(nn.Linear(hidden_size, 1), nn.Sigmoid()), nn.Sigmoid()
        )
        self.phi = Categorical(d * action_spaces.a.n * int(self.subtask_nvec.prod()), 2)

        self.zeta = Categorical(self.line_size, len(LineTypes._fields))

        # NOTE {
        self.phi_debug = nn.Sequential(
            init_(nn.Linear(1, 1), nn.Sigmoid()), nn.Sigmoid()
        )
        self.xi_debug = nn.Sequential(
            Parallel(
                nn.Sequential(Reshape(1, d, h, w)),
                nn.Sequential(Reshape(self.line_size, 1, 1, 1)),
            ),
            Product(),
            Times(
                100 * (F.pad(torch.eye(4), (1, 2, 15, 0)).view(1, 19, 7, 1, 1) - 0.5)
            ),
            Reshape(self.line_size * d, h * w),
            # init_(nn.Conv2d(d * self.line_size, hidden_size, kernel_size=1), "sigmoid"),
            # Reshape(hidden_size, h * w),
            Sum(dim=-1),
            activation,
            Times(100),
            Sum(dim=-1, keepdim=True),
        )
        self.zeta_debug = Categorical(len(LineTypes._fields), len(LineTypes._fields))
        # NOTE }

        input_size = h * w * hidden_size  # conv output
        if isinstance(action_spaces.a, Discrete):
            num_outputs = action_spaces.a.n
            self.actor = Categorical(input_size, num_outputs)
        elif isinstance(action_spaces.a, Box):
            num_outputs = action_spaces.a.shape[0]
            self.actor = DiagGaussian(input_size, num_outputs)
        else:
            raise NotImplementedError

        self.critic = init_(nn.Linear(input_size, 1))

        state_sizes = RecurrentState(
            a=1,
            g=1,
            cg=1,
            cr=1,
            l=1,
            z=self.n_subtasks,
            a_probs=action_spaces.a.n,
            g_probs=self.n_subtasks,
            cg_probs=2,
            cr_probs=2,
            l_probs=2,
            z_probs=self.n_subtasks * len(LineTypes._fields),
            r=self.line_size,
            p=self.n_subtasks,
            v=1,
            last_condition=self.line_size,
            last_eval=1,
        )
        self.state_sizes = RecurrentState(*map(int, state_sizes))

        # embeddings
        def eye_embedding(n):
            return nn.Embedding.from_pretrained(torch.eye(int(n)))

        self.g_discrete_one_hots = nn.ModuleList(
            [eye_embedding(n) for n in self.subtask_nvec]
        )
        self.a_one_hots = eye_embedding(n=action_spaces.a.n)
        self.g_one_hots = eye_embedding(action_spaces.g.n)
        self.z_one_hots = eye_embedding(len(LineTypes()))
        self.p_one_hots = eye_embedding(self.n_subtasks)

        # buffers
        one_step = F.pad(torch.eye(self.n_subtasks - 1), [1, 0, 0, 1])
        one_step[:, -1] += 1 - one_step.sum(-1)
        self.register_buffer("one_step", one_step.unsqueeze(0))
        no_op_probs = torch.zeros(1, self.actor.linear.out_features)
        no_op_probs[:, -1] = 1
        self.register_buffer("no_op_probs", no_op_probs)

    def print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def parse_hidden(self, hx):
        return RecurrentState(*torch.split(hx, self.state_sizes, dim=-1))

    def parse_inputs(self, inputs):
        return Obs(*torch.split(inputs, self.obs_sections, dim=-1))

    def forward(self, inputs, rnn_hxs):
        T, N, D = inputs.shape

        # {{{
        # detach actions
        # noinspection PyProtectedMember
        inputs, *actions = torch.split(
            inputs.detach(), [D - sum(self.size_actions)] + self.size_actions, dim=-1
        )
        actions = Actions(*actions)

        # parse non-action inputs
        inputs = self.parse_inputs(inputs)
        inputs = inputs._replace(base=inputs.base.view(T, N, *self.obs_shape))

        # build memory
        task = inputs.subtasks.view(
            *inputs.subtasks.shape[:2], self.n_subtasks, self.subtask_nvec.size
        )[0]
        task_columns = torch.split(task, 1, dim=-1)
        g_discrete = [x.squeeze(2) for x in task_columns]
        M = g_discrete_to_binary(g_discrete, self.g_discrete_one_hots.children())
        new_episode = torch.all(rnn_hxs == 0, dim=-1).squeeze(0)

        # NOTE {
        debug_in = M[:, :, -self.subtask_nvec[-2:].sum() : -self.subtask_nvec[-1]]
        # M_zeta = self.zeta_debug(debug_in)
        truth = FixedCategorical(probs=debug_in)
        # M_zeta_dist = self.zeta_debug(debug_in)
        M_zeta_dist = self.zeta(M)
        # self.print("M_zeta_dist.probs")
        # self.print(round(M_zeta_dist.probs, 2))
        # M_zeta_dist = truth
        z = actions.z[0].long()  # use time-step 0; z fixed throughout episode
        self.sample_new(z, M_zeta_dist)
        # NOTE }
        hx = self.parse_hidden(rnn_hxs)
        for x in hx:
            x.squeeze_(0)
        hx.p[new_episode, 0] = 1.0  # initialize pointer to first subtask
        hx.r[new_episode] = M[new_episode, 0]  # initialize r to first subtask
        # initialize g to first subtask
        hx.g[new_episode] = 0.0
        hx.z[new_episode] = z[new_episode].float()
        hx.z_probs[new_episode] = M_zeta_dist.probs.view(
            -1, self.n_subtasks * len(LineTypes._fields)
        )[new_episode]

        return self.pack(
            self.inner_loop(
                inputs=inputs,
                hx=hx,
                M=M,
                M_discrete=task,
                subtask=inputs.subtask,
                actions=actions,
            )
        )

    def pack(self, outputs):
        zipped = list(zip(*outputs))
        # for name, x in zip(RecurrentState._fields, zipped):
        #     if not x:
        #         print(name)

        stacked = [torch.stack(x).float() for x in zipped]
        preprocessed = [x.view(*x.shape[:2], -1) for x in stacked]

        # for name, x, size in zip(
        #     RecurrentState._fields, preprocessed, self.state_sizes
        # ):
        #     if x.size(2) != size:
        #         print(name, x, size)
        #     if x.dtype != torch.float32:
        #         print(name)

        hx = torch.cat(preprocessed, dim=-1)
        return hx, hx[-1:]

    def inner_loop(
        self, hx: RecurrentState, actions: Actions, inputs: Obs, M, M_discrete, subtask
    ):
        _, N, *_ = inputs.base.shape
        p = hx.p
        T = LineTypes()

        # combine past and present actions (sampled values)
        obs = inputs.base

        def action_vector(current, previous):
            return torch.cat([current, previous.unsqueeze(0)], dim=0).long().squeeze(2)

        A = action_vector(actions.a, hx.a)
        G = action_vector(actions.g, hx.g)
        L = action_vector(actions.l, hx.l)
        CR = action_vector(actions.cr, hx.cr)
        CG = action_vector(actions.cg, hx.cg)
        M_zeta = self.z_one_hots(hx.z.long())

        for t in range(inputs.base.shape[0]):
            self.print(T)
            self.print("M_zeta")
            for _z in M_zeta[0]:
                self.print(T._fields[int(_z.argmax())])

            def safediv(x, y):
                return torch.clamp(x / torch.clamp(y, min=1e-5), 0.0, 1.0)

            # e
            e = (p.unsqueeze(1) @ M_zeta).permute(2, 0, 1)

            # condition
            condition = interp(
                hx.r,
                hx.last_condition,
                safediv(e[T.EndWhile], e[[T.If, T.While, T.EndWhile]].sum(0)),
            )

            x = self.xi((inputs.base[t], hx.r))
            l = self.dist(x)
            # self.sample_new(L[t], l_dist)
            # l = L[t, :1].float()

            # l
            # xi_in = self.xi_debug((inputs.base[t], condition)).detach()
            # self.print("l", round(l, 4))
            # NOTE {
            # c = torch.split(condition, list(self.subtask_nvec), dim=-1)[-1][:, 1:]
            # last_condition = torch.split(
            # hx.last_condition, list(self.subtask_nvec), dim=-1
            # )[-1][:, 1:]
            # hx_r = torch.split(hx.r, list(self.subtask_nvec), dim=-1)[-1][:, 1:]
            # self.print("last_condition", last_condition)
            # self.print("r", hx_r)
            # self.print("l condition", c)
            # phi_in = inputs.base[t, :, 1:-2] * c.view(N, -1, 1, 1)
            # truth = torch.max(phi_in.view(N, -1), dim=-1).values.float().view(N, 1)

            # self.print("l truth", round(truth, 4))
            # l = truth
            # NOTE }
            # self.print("xi_in", xi_in)
            # l = self.xi(xi_in)
            # self.print("l1", l)

            # control memory
            last_eval = interp(hx.last_eval, l, e[T.If])
            last_condition = interp(hx.last_condition, hx.r, e[T.While])

            # l'
            l = interp(
                l,
                1 - hx.last_eval,
                safediv(e[T.Else], e[[T.If, T.Else, T.While, T.EndWhile]].sum(0)),
            )
            # self.print("l2", l)
            l_probs = torch.cat([1 - l, l], dim=1)
            l_dist = FixedCategorical(probs=l_probs)
            self.print("l2", l_dist.probs)
            self.sample_new(L[t], l_dist)
            l = L[t].unsqueeze(-1).float()

            def roll(x):
                return F.pad(x, [1, 0])[:, :-1]

            def scan(*idxs, cumsum, it):
                p = []
                omega = M_zeta[:, :, idxs].sum(-1) * cumsum
                *it, last = it
                for i in it:
                    p.append((1 - sum(p)) * omega[:, i])
                p.append(1 - sum(p))
                return torch.stack(p, dim=-1)

            # p
            scan_forward = functools.partial(
                scan, cumsum=roll(torch.cumsum(hx.p, dim=-1)), it=range(M.size(1))
            )
            scan_backward = functools.partial(
                scan,
                cumsum=roll(torch.cumsum(hx.p.flip(-1), dim=-1)).flip(-1),
                it=range(M.size(1) - 1, -1, -1),
            )
            p_step = (p.unsqueeze(1) @ self.one_step).squeeze(1)
            self.print("cr before update", round(hx.cr, 2))
            pIf = interp(scan_forward(T.Else, T.EndIf), p_step, l)
            pElse = interp(scan_forward(T.EndIf), p_step, l)
            pWhile = interp(scan_forward(T.EndWhile), p_step, l)
            pEndWhile = interp(p_step, scan_backward(T.While).flip(-1), l)
            pSubtask = interp(hx.p, p_step, hx.cr)
            self.print("p before update", round(p, 2))
            p = (
                # e[[L.If, L.While, L.Else]].sum(0)  # conditions
                # * interp(scan_forward(L.EndIf, L.Else, L.EndWhile), p_step, l)
                e[T.If] * pIf
                + e[T.Else] * pElse
                + e[T.While] * pWhile
                + e[T.EndWhile] * pEndWhile
                + e[T.EndIf] * p_step
                + e[T.Subtask] * pSubtask
            )
            is_line = 1 - inputs.ignore[0]
            p = torch.clamp(p, 0.0, 1.0)
            p = is_line * p / p.sum(-1, keepdim=True)  # zero out non-lines

            # concentrate non-allocated attention on last line
            last_line = is_line.sum(-1).long() - 1
            p = p + (1 - p.sum(-1, keepdim=True)) * self.p_one_hots(last_line)

            self.print("e[L.If]", e[T.If])
            self.print("e[L.Else]", e[T.Else])
            self.print("e[L.EndIf]", e[T.EndIf])
            self.print("e[L.While]", e[T.While])
            self.print("e[L.EndWhile]", e[T.EndWhile])
            self.print("e[L.Subtask]", e[T.Subtask])

            # r
            r = (p.unsqueeze(1) @ M).squeeze(1)

            # g
            old_g = self.g_one_hots(G[t - 1])
            cg = e[T.Subtask] * hx.cg + (1 - e[T.Subtask])
            probs = interp(old_g, p, cg)
            g_dist = FixedCategorical(probs=torch.clamp(probs, 0.0, 1.0))
            self.sample_new(G[t], g_dist)

            # a
            g = G[t]
            g_binary = M[torch.arange(N), g]
            conv_out = self.conv2((obs[t], broadcast3d(g_binary, self.obs_shape[1:])))
            if self.agent is None:
                probs = self.actor(conv_out).dist.probs
            else:
                agent_inputs = ppo.control_flow.lower_level.Obs(
                    base=obs[t].view(N, -1),
                    subtask=g_binary[:, : self.agent_subtask_size],
                )
                probs = self.agent(
                    agent_inputs, z=hx.z.long(), rnn_hxs=None, masks=None
                ).dist.probs
            op = g_binary[:, self.agent_subtask_size].unsqueeze(1)
            no_op = 1 - op
            a_dist = FixedCategorical(
                # if subtask is a control-flow statement, force no-op
                probs=op * probs
                + no_op * self.no_op_probs.expand(op.size(0), -1)
            )
            self.sample_new(A[t], a_dist)

            # a[:] = 'wsadeq'.index(input('act:'))
            self.print("p after update", round(p, 2))

            def gating_function(subtask_param, C):
                task_sections = torch.split(
                    subtask_param, tuple(self.subtask_nvec), dim=-1
                )
                parts = (self.conv1(obs[t]), self.a_one_hots(A[t])) + task_sections
                outer_product_obs = 1
                for i1, part in enumerate(parts):
                    for i2 in range(len(parts)):
                        if i1 != i2:
                            part.unsqueeze_(i2 + 1)
                    outer_product_obs = outer_product_obs * part

                c_dist = self.phi(outer_product_obs.view(N, -1))
                # self.sample_new(C[t], c_dist)

                # NOTE {
                # _task_sections = torch.split(
                #     subtask_param, tuple(self.subtask_nvec), dim=-1
                # )
                # interaction, count, obj, _, condition = _task_sections
                # agent_layer = obs[t, :, 6, :, :].long()
                # j, k, l = torch.split(agent_layer.nonzero(), 1, dim=-1)
                # debug_obs = obs[t, j, :, k, l].squeeze(1)
                # a_one_hot = self.a_one_hots(A[t])
                # correct_object = obj * debug_obs[:, 1 : 1 + self.subtask_nvec[2]]
                # column1 = interaction[:, :1]
                # column2 = interaction[:, 1:] * a_one_hot[:, 4:-1]
                # correct_action = torch.cat([column1, column2], dim=-1)
                # truth = (
                #     correct_action.sum(-1, keepdim=True)
                #     * correct_object.sum(-1, keepdim=True)
                # ).detach()  # * condition[:, :1] + (1 - condition[:, :1])
                # # c = self.phi_debug(truth)
                # c = truth
                # self.print("c", round(c, 4))
                # NOTE }
                return c_dist.probs[:, 1:], c_dist.probs

            # cr
            cr, cr_probs = gating_function(subtask_param=r, C=CR)

            # cg
            g = M[torch.arange(N), G[t]]
            cg, cg_probs = gating_function(subtask_param=g, C=CG)

            yield RecurrentState(
                cg=cg,
                cr=cr,
                cg_probs=cg_probs,
                cr_probs=cr_probs,
                l=L[t],
                p=p,
                r=r,
                g=G[t],
                g_probs=g_dist.probs,
                a=A[t],
                a_probs=a_dist.probs,
                v=self.critic_linear(x),
                last_condition=last_condition,
                last_eval=last_eval,
                z=hx.z,
                z_probs=hx.z_probs,
                l_probs=l_dist.probs,
            )

    @property
    def is_recurrent(self):
        return False

    @staticmethod
    def sample_new(x, dist):
        new = x < 0
        x[new] = dist.sample()[new].flatten()
