from collections import namedtuple
import itertools

from gym.spaces import Box, Discrete
import numpy as np
import torch
from torch import nn as nn
import torch.jit
from torch.nn import functional as F

from gridworld_env.control_flow_gridworld import LineTypes
from gridworld_env.subtasks_gridworld import Obs
import ppo
from ppo.distributions import Categorical, DiagGaussian, FixedCategorical
from ppo.layers import Concat, Flatten, Parallel, Product, Reshape, ShallowCopy, Sum
from ppo.subtasks import Actions, Teacher
from ppo.subtasks.teacher import g_binary_to_discrete, g_discrete_to_binary
from ppo.utils import broadcast3d, init_, interp, trace

RecurrentState = namedtuple(
    "RecurrentState",
    "a cg cr r p g a_probs cg_probs cr_probs g_probs v g_loss subtask P",
)


def sample_new(x, dist):
    new = x < 0
    x[new] = dist.sample()[new].flatten()


class Recurrence(torch.jit.ScriptModule):
    __constants__ = ["input_sections", "subtask_space", "state_sizes", "recurrent"]

    def __init__(
        self,
        obs_spaces,
        action_spaces,
        hidden_size,
        recurrent,
        hard_update,
        agent,
        multiplicative_interaction,
    ):
        super().__init__()
        self.hard_update = hard_update
        self.multiplicative_interaction = multiplicative_interaction
        if agent:
            assert isinstance(agent, Teacher)
        self.agent = agent
        self.recurrent = recurrent
        self.obs_spaces = obs_spaces
        self.n_subtasks = self.obs_spaces.subtasks.nvec.shape[0]
        self.subtask_nvec = self.obs_spaces.subtasks.nvec[0]
        d, h, w = self.obs_shape = obs_spaces.base.shape
        self.obs_sections = [int(np.prod(s.shape)) for s in self.obs_spaces]

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
            init_(
                nn.Conv2d(d + int(self.subtask_nvec.sum()), hidden_size, kernel_size=1),
                "relu",
            ),
            nn.ReLU(),
            Flatten(),
        )

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

        self.phi = trace(
            lambda in_size: init_(nn.Linear(in_size, 2), "sigmoid"),
            in_size=(d * action_spaces.a.n * int(self.subtask_nvec.prod())),
        )
        self.phi_update2 = init_(nn.Linear(int(self.subtask_nvec.sum()), 1), "sigmoid")

        for i, x in enumerate(self.subtask_nvec):
            self.register_buffer(f"part{i}_one_hot", torch.eye(int(x)))
        self.register_buffer("a_one_hots", torch.eye(int(action_spaces.a.n)))
        self.register_buffer("g_one_hots", torch.eye(action_spaces.g.n))
        self.register_buffer(
            "subtask_space", torch.tensor(self.subtask_nvec.astype(np.int64))
        )

        trans = F.pad(torch.eye(self.n_subtasks), [1, 0])[:, :-1]
        trans[-1, -1] = 1

        self.register_buffer("trans", trans)

        state_sizes = RecurrentState(
            a=1,
            cg=1,
            cr=1,
            r=int(self.subtask_nvec.sum()),
            p=self.n_subtasks,
            g=1,
            a_probs=action_spaces.a.n,
            cg_probs=2,
            cr_probs=2,
            g_probs=self.n_subtasks,
            v=1,
            g_loss=1,
            subtask=1,
            P=self.obs_spaces.subtasks.nvec[0, -1]
            + 1,  # +1 for previous evaluation of condition
        )
        self.state_sizes = RecurrentState(*map(int, state_sizes))

        self.line_size = int(self.obs_spaces.subtasks.nvec[0].sum())
        h_size = d * self.line_size

        self.xi = nn.Sequential(
            Parallel(
                nn.Sequential(Reshape(1, d, h, w)),
                nn.Sequential(Reshape(self.line_size, 1, 1, 1)),
            ),
            Product(),
            Reshape(d * self.line_size, *self.obs_shape[-2:]),
            # init_(
            # nn.Conv2d(self.condition_size * d, hidden_size, kernel_size=1, stride=1)
            # ),
            # attention {
            ShallowCopy(2),
            Parallel(
                Reshape(h_size, h * w),
                nn.Sequential(
                    init_(nn.Conv2d(h_size, 1, kernel_size=1)),
                    Reshape(1, h * w),
                    nn.Softmax(dim=-1),
                ),
            ),
            Product(),
            Sum(dim=-1),
            # }
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(h_size, 1), "sigmoid"),
            # init_(nn.Linear(d * self.condition_size * 4 * 4, 1), "sigmoid"),
            nn.Sigmoid(),
            Reshape(1, 1),
        )

        # buffers
        one_step = F.pad(torch.eye(self.n_subtasks - 1), [1, 0, 0, 1])
        one_step[:, -1] += 1 - one_step.sum(-1)
        self.register_buffer("one_step", one_step.unsqueeze(0))
        self.register_buffer(
            f"part3_one_hot", torch.eye(int(self.obs_spaces.subtasks.nvec[0, -1]))
        )
        no_op_probs = torch.zeros(1, self.actor.linear.out_features)
        no_op_probs[:, -1] = 1
        self.register_buffer("no_op_probs", no_op_probs)
        self.register_buffer("last_line", torch.eye(self.obs_spaces.nvec[0].sum()))

        self.size_agent_subtask = int(self.obs_spaces.subtasks.nvec[0, :-1].sum())
        self.agent_input_size = int(self.obs_spaces.subtasks.nvec[0, :-1].sum())
        self.zeta = nn.Sequential(
            init_(nn.Linear(self.line_size, len(LineTypes._fields))), nn.Softmax()
        )
        self.state_sizes = RecurrentState(
            **self.state_sizes._asdict(),
            P=self.obs_spaces.subtasks.nvec[0, -1]
            + 1,  # +1 for previous evaluation of condition
        )

    # @torch.jit.script_method
    def parse_hidden(self, hx):
        return RecurrentState(*torch.split(hx, self.state_sizes, dim=-1))

    # @torch.jit.script_method
    def g_binary_to_int(self, g_binary):
        g123 = g_binary_to_discrete(g_binary, self.subtask_nvec)
        g123[:, :-1] *= self.subtask_nvec[1:]  # g1 * x2, g2 * x3
        g123[:, 0] *= self.subtask_nvec[2]  # g1 * x3
        return g123.sum(dim=-1)

    # @torch.jit.script_method
    def g_int_to_123(self, g):
        x1, x2, x3 = self.subtask_nvec.to(g.dtype)
        g1 = g // (x2 * x3)
        x4 = g % (x2 * x3)
        g2 = x4 // x3
        g3 = x4 % x3
        return g1, g2, g3

    @property
    def g_discrete_one_hots(self):
        for i in itertools.count():
            try:
                yield getattr(self, f"part{i}_one_hot")
            except AttributeError:
                break

    def parse_inputs(self, inputs):
        return Obs(*torch.split(inputs, self.obs_sections, dim=2))

    def get_a_dist(self, conv_out, g_binary, obs):
        N = obs.size(0)
        if self.agent is None:
            return self.actor(conv_out)
        else:
            agent_inputs = ppo.subtasks.teacher.Obs(
                base=obs.view(N, -1), subtask=g_binary
            )
            return self.agent(agent_inputs, rnn_hxs=None, masks=None).dist

    # @torch.jit.script_method
    def forward(self, inputs, hx):
        assert hx is not None
        T, N, D = inputs.shape

        # detach actions
        # noinspection PyProtectedMember
        n_actions = len(Actions._fields)
        inputs, *actions = torch.split(
            inputs.detach(), [D - n_actions] + [1] * n_actions, dim=2
        )
        actions = Actions(*actions)

        # parse non-action inputs
        inputs = self.parse_inputs(inputs)
        inputs = inputs._replace(base=inputs.base.view(T, N, *self.obs_shape))

        # build memory
        task = inputs.subtasks.view(
            *inputs.subtasks.shape[:2], self.n_subtasks, self.subtask_nvec.size
        )
        task = torch.split(task, 1, dim=-1)
        g_discrete = [x[0, :, :, 0] for x in task]
        M_discrete = torch.stack(g_discrete, dim=-1)

        M = g_discrete_to_binary(g_discrete, self.g_discrete_one_hots)

        # parse hidden
        new_episode = torch.all(hx.squeeze(0) == 0, dim=-1)
        hx = self.parse_hidden(hx)
        p = hx.p
        r = hx.r
        for x in hx:
            x.squeeze_(0)
        p[new_episode, 0] = 1.0  # initialize pointer to first subtask
        r[new_episode] = M[new_episode, 0]  # initialize r to first subtask
        # initialize g to first subtask
        hx.g[new_episode] = 0.0

        return self.pack(
            self.inner_loop(
                inputs=inputs,
                hx=hx,
                M=M,
                M_discrete=M_discrete,
                N=N,
                T=T,
                subtask=inputs.subtask,
                p=p,
                r=r,
                actions=actions,
            )
        )

    def inner_loop(self, hx, M, M_discrete, N, T, subtask, p, r, actions, inputs):
        subtask = subtask.long().view(T, N)
        M_zeta = self.zeta(M).transpose().unsqueeze(-1)
        is_line = torch.any(M > 0, dim=-1).float()
        non_line = 1 - is_line
        last_line = self.last_line[is_line.sum(-1)]

        # NOTE {
        # TODO: truth
        # NOTE }
        L = LineTypes()

        # combine past and present actions (sampled values)
        obs = inputs.base
        A = torch.cat([actions.a, hx.a.unsqueeze(0)], dim=0).long().squeeze(2)
        G = torch.cat([actions.g, hx.g.unsqueeze(0)], dim=0).long().squeeze(2)
        for t in range(T):

            # e
            e = (p.unsqueeze(1) @ M_zeta).squeeze(1)
            eP = e[L.If, L.While].sum(-1, keepdim=True)
            er = eP / e[L.If, L.Else, L.While, L.EndWhile].sum(-1, keepdim=True)

            # l
            # NOTE {
            # TODO: truth
            # NOTE }
            l = self.xi((inputs.base[t], interp(hx.P, hx.r, er)))

            # P
            P = (
                e[L.If] * F.pad(l, [0, self.line_size])  # record evaluation
                + e[L.While] * F.pad(r, [1, 0])  # record condition
                + (1 - e[L.If] - e[L.While]) * hx.P  # keep the same
            )

            def scan(*idxs, u, it):
                p = []
                for i in it:
                    p.append((1 - sum(p) * M_zeta[idxs].sum(-1) * u))
                return p

            # p
            p_forward = scan(
                L.EndIf,
                L.Else,
                L.EndWhile,
                u=torch.cumsum(hx.p, dim=-1),
                it=range(M.size(1)),
            )
            p_backward = scan(
                L.While,
                u=torch.cumsum(hx.p.flip(-1), dim=-1).flip(-1),
                it=range(M.size(1) - 1, -1, -1),
            )
            p_step = (p.unsqueeze(1) @ self.one_step).squeeze(1)
            p = (
                e[L.If, L.While, L.Else].sum(-1)  # conditions
                * (l * p_step + (1 - l) * p_forward)
                + e[L.EndWhile] * (l * p_backward + (1 - l) * p_step)
                + e[L.EndIf] * p_step
                + e[L.Subtask] * (hx.cr * p_step + (1 - hx.cr) * hx.p)
            )
            overflow_attention = torch.sum(non_line * p, dim=-1)
            p = is_line * p + overflow_attention * last_line

            # r
            r = (p.unsqueeze(1) @ M).squeeze(1)

            # g
            old_g = self.g_one_hots[G[t - 1]]
            probs = interp(old_g, p, hx.cg)
            g_dist = FixedCategorical(probs=torch.clamp(probs, 0.0, 1.0))
            sample_new(G[t], g_dist)

            # a
            g = G[t]
            g_binary = M[torch.arange(N), g]
            conv_out = self.conv2((obs[t], broadcast3d(g_binary, self.obs_shape[1:])))
            probs = (
                super()
                .get_a_dist(conv_out, g_binary[:, : self.agent_input_size], obs)
                .probs
            )
            op = g_binary[:, self.agent_input_size].unsqueeze(1)
            no_op = 1 - op
            a_dist = FixedCategorical(
                # if subtask is a control-flow statement, force no-op
                probs=op * probs
                + no_op * self.no_op_probs.expand(op.size(0), -1)
            )
            sample_new(A[t], a_dist)
            # a[:] = 'wsadeq'.index(input('act:'))

            def gating_function(subtask_param):
                task_sections = torch.split(
                    subtask_param, tuple(self.subtask_nvec), dim=-1
                )
                # NOTE {
                # agent_layer = obs[t, :, 6, :, :].long()
                # j, k, l = torch.split(agent_layer.nonzero(), 1, dim=-1)
                # debug_obs = obs[t, j, :, k, l].squeeze(1)
                # a_one_hot = self.a_one_hots[A[t]]
                # interaction, count, obj, condition = task_sections
                # correct_object = obj * debug_obs[:, 1 : 1 + self.subtask_nvec[2]]
                # column1 = interaction[:, :1]
                # column2 = interaction[:, 1:] * a_one_hot[:, 4:-1]
                # correct_action = torch.cat([column1, column2], dim=-1)
                # truth = (
                #     correct_action.sum(-1, keepdim=True)
                #     * correct_object.sum(-1, keepdim=True)
                # ).detach()  # * condition[:, :1] + (1 - condition[:, :1])
                # NOTE }
                parts = (self.conv1(obs[t]), self.a_one_hots[A[t]]) + task_sections
                outer_product_obs = 1
                for i1, part in enumerate(parts):
                    for i2 in range(len(parts)):
                        if i1 != i2:
                            part.unsqueeze_(i2 + 1)
                    outer_product_obs = outer_product_obs * part

                c_logits = self.phi(outer_product_obs.view(N, -1))
                if self.hard_update:
                    raise NotImplementedError
                    # c_dist = FixedCategorical(logits=c_logits)
                    # c = actions.c[t]
                    # sample_new(c, c_dist)
                    # probs = c_dist.probs
                else:
                    c = torch.sigmoid(c_logits[:, :1])
                    probs = torch.zeros_like(c_logits)  # dummy value
                return c, probs

            # cr
            cr, cr_probs = gating_function(subtask_param=r)

            # cg
            g = M[torch.arange(N), G[t]]
            cg, cg_probs = gating_function(subtask_param=g)

            yield RecurrentState(
                cg=cg,
                cr=cr,
                cg_probs=cg_probs,
                cr_probs=cr_probs,
                p=p,
                r=r,
                g=G[t],
                g_probs=g_dist.probs,
                g_loss=-g_dist.log_probs(subtask[t]),
                a=A[t],
                a_probs=a_dist.probs,
                subtask=subtask[t],
                v=self.critic(conv_out),
                P=P,
            )

    def pack(self, outputs):
        zipped = list(zip(*outputs))
        # for name, x in zip(RecurrentState._fields, zipped):
        #    if not x:
        #        print(name)
        #        import ipdb

        #        ipdb.set_trace()

        stacked = [torch.stack(x) for x in zipped]
        preprocessed = [x.float().view(*x.shape[:2], -1) for x in stacked]

        # for name, x, size in zip(
        #    RecurrentState._fields, preprocessed, self.state_sizes
        # ):
        #    if x.size(2) != size:
        #        print(name, x, size)
        #        import ipdb

        #        ipdb.set_trace()
        #    if x.dtype != torch.float32:
        #        print(name)
        #        import ipdb

        #        ipdb.set_trace()

        hx = torch.cat(preprocessed, dim=-1)
        return hx, hx[-1]
