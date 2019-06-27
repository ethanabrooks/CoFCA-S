from collections import namedtuple

from gym.spaces import Box, Discrete
import numpy as np
import torch
from torch import nn as nn
import torch.jit
from torch.nn import functional as F

import ppo
from ppo.agent import AgentValues, NNBase
from ppo.distributions import Categorical, DiagGaussian, FixedCategorical
from ppo.layers import Concat, Flatten, Parallel, Product
import ppo.subtasks.teacher
from ppo.subtasks.teacher import Teacher, g123_to_binary, g_binary_to_123
from ppo.subtasks.wrappers import Actions, Obs
from ppo.utils import broadcast3d, init_, interp, trace

RecurrentState = namedtuple(
    'RecurrentState',
    'a cg cr r p g a_probs cg_probs cr_probs g_probs v g_loss cr_loss cg_loss subtask')


# noinspection PyMissingConstructor
class Agent(ppo.agent.Agent, NNBase):
    def __init__(self, obs_spaces, action_space, hidden_size, entropy_coef, hard_update, agent,
                 **kwargs):
        nn.Module.__init__(self)
        self.hard_update = hard_update
        self.entropy_coef = entropy_coef
        self.action_spaces = Actions(*action_space.spaces)
        self.obs_spaces = obs_spaces
        self.recurrent_module = self.build_recurrent_module(
            agent=agent,
            hard_update=hard_update,
            hidden_size=hidden_size,
            obs_spaces=self.obs_spaces,
            action_spaces=self.action_spaces,
            **kwargs)
        self.agent = agent

    # noinspection PyMethodOverriding
    def build_recurrent_module(self, agent, hard_update, hidden_size, **kwargs):
        return Recurrence(
            hidden_size=hidden_size,
            hard_update=hard_update,
            agent=agent,
            **kwargs,
        )

    def forward(self, inputs, rnn_hxs, masks, action=None, deterministic=False):
        n = inputs.size(0)
        actions = None
        if action is not None:
            actions = Actions(*torch.split(action, [1] * len(self.action_spaces), dim=-1))

        all_hxs, last_hx = self._forward_gru(inputs.view(n, -1), rnn_hxs, masks, actions=actions)
        rm = self.recurrent_module
        hx = RecurrentState(*rm.parse_hidden(all_hxs))

        if action is None:
            actions = Actions(a=hx.a, cg=hx.cg, cr=hx.cr, g=hx.g)

        if self.hard_update:
            dists = Actions(
                a=FixedCategorical(hx.a_probs),
                cg=FixedCategorical(hx.cg_probs),
                cr=FixedCategorical(hx.cr_probs),
                g=FixedCategorical(hx.g_probs),
            )
        else:
            dists = Actions(
                a=None if self.agent else FixedCategorical(hx.a_probs),
                cg=None,
                cr=None,
                g=FixedCategorical(hx.g_probs),
            )

        log_probs = sum(dist.log_probs(a) for dist, a in zip(dists, actions) if dist is not None)
        entropies = sum(dist.entropy() for dist in dists if dist is not None)
        aux_loss = -self.entropy_coef * entropies.mean()
        log = {k: v for k, v in hx._asdict().items() if k.endswith('_loss')}

        return AgentValues(
            value=hx.v,
            action=torch.cat(actions, dim=-1),
            action_log_probs=log_probs,
            aux_loss=aux_loss,
            rnn_hxs=torch.cat(hx, dim=-1),
            dist=None,
            log=log)

    def get_value(self, inputs, rnn_hxs, masks):
        n = inputs.size(0)
        all_hxs, last_hx = self._forward_gru(inputs.view(n, -1), rnn_hxs, masks)
        return self.recurrent_module.parse_hidden(all_hxs).v

    def _forward_gru(self, x, hxs, masks, actions=None):
        if actions is None:
            y = F.pad(x, [0, len(Actions._fields)], 'constant', -1)
        else:
            y = torch.cat([x] + list(actions), dim=-1)
        return super()._forward_gru(y, hxs, masks)

    @property
    def recurrent_hidden_state_size(self):
        return sum(self.recurrent_module.state_sizes)

    @property
    def is_recurrent(self):
        return True


def sample_new(x, dist):
    new = x < 0
    x[new] = dist.sample()[new].flatten()


class Recurrence(torch.jit.ScriptModule):
    __constants__ = ['input_sections', 'subtask_space', 'state_sizes', 'recurrent']

    def __init__(self, obs_spaces, action_spaces, hidden_size, recurrent, hard_update, agent,
                 multiplicative_interaction):
        super().__init__()
        self.hard_update = hard_update
        self.multiplicative_interaction = multiplicative_interaction
        if agent:
            assert isinstance(agent, Teacher)
        self.agent = agent
        self.recurrent = recurrent
        self.obs_spaces = obs_spaces
        self.subtask_nvec = self.obs_spaces.subtasks.nvec[0]
        d, h, w = self.obs_shape = obs_spaces.base.shape
        self.obs_sections = self.get_obs_sections()

        self.conv = nn.Sequential(
            Concat(dim=1),
            init_(
                nn.Conv2d(
                    d + int(self.subtask_nvec.sum()),
                    hidden_size,
                    kernel_size=3,  # TODO: kernel_size=1
                    stride=1,
                    padding=1),
                'relu'),
            nn.ReLU(),
            Flatten())

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

        if multiplicative_interaction:
            self.phi_update = nn.Sequential(
                Parallel(
                    init_(nn.Linear(d, hidden_size)),  # obs
                    init_(nn.Linear(action_spaces.a.n, hidden_size)),  # action
                    *[init_(nn.Linear(i, hidden_size))
                      for i in self.subtask_nvec],  # subtask parameter
                ),
                Product(),
                init_(nn.Linear(hidden_size, 2), 'sigmoid'))
        else:
            self.phi_update = trace(
                lambda in_size: init_(nn.Linear(in_size, 2), 'sigmoid'),
                in_size=(d * action_spaces.a.n * int(self.subtask_nvec.prod())))

        for i, x in enumerate(self.subtask_nvec):
            self.register_buffer(f'part{i}_one_hot', torch.eye(int(x)))
        self.register_buffer('a_one_hots', torch.eye(int(action_spaces.a.n)))
        self.register_buffer('g_one_hots', torch.eye(action_spaces.g.n))
        self.register_buffer('subtask_space', torch.tensor(self.subtask_nvec.astype(np.int64)))

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
            cr_loss=1,
            cg_loss=1,
            subtask=1,
        )
        self.state_sizes = RecurrentState(*map(int, state_sizes))
        self.register_agent_dummy_values()

    def register_agent_dummy_values(self):
        self.register_buffer(
            'agent_dummy_values',
            torch.zeros(1, self.obs_sections.subtasks + self.obs_sections.next_subtask))

    @property
    def n_subtasks(self):
        return self.obs_spaces.subtasks.nvec.shape[0]

    def get_obs_sections(self):
        return Obs(*[int(np.prod(s.shape)) for s in self.obs_spaces])

    # @torch.jit.script_method
    def parse_hidden(self, hx):
        return RecurrentState(*torch.split(hx, self.state_sizes, dim=-1))

    # @torch.jit.script_method
    def g_binary_to_int(self, g_binary):
        g123 = g_binary_to_123(g_binary, self.subtask_nvec)
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

    def check_grad(self, **kwargs):
        for k, v in kwargs.items():
            if v.grad_fn is not None:
                grads = torch.autograd.grad(
                    v.mean(), self.parameters(), retain_graph=True, allow_unused=True)
                for (name, _), grad in zip(self.named_parameters(), grads):
                    if grad is None:
                        print(f'{k} has no grad wrt {name}')
                    else:
                        print(f'mean grad ({v.mean().item()}) of {k} wrt {name}:', grad.mean())
                        if torch.isnan(grad.mean()):
                            import ipdb
                            ipdb.set_trace()

    # @torch.jit.script_method
    def forward(self, inputs, hx):
        assert hx is not None
        T, N, D = inputs.shape

        # detach actions
        # noinspection PyProtectedMember
        n_actions = len(Actions._fields)
        inputs, *actions = torch.split(inputs.detach(), [D - n_actions] + [1] * n_actions, dim=2)
        actions = Actions(*actions)

        # parse non-action inputs
        inputs = Obs(*torch.split(inputs, self.obs_sections, dim=2))
        obs = inputs.base.view(T, N, *self.obs_shape)
        task = inputs.subtasks.view(T, N, self.n_subtasks, self.subtask_nvec.size)

        # build memory
        task = torch.split(task, 1, dim=-1)
        interaction, count, obj = [x[0, :, :, 0] for x in task]
        M123 = torch.stack([interaction, count, obj], dim=-1)
        one_hots = [self.part0_one_hot, self.part1_one_hot, self.part2_one_hot]
        g123 = (interaction, count, obj)
        M = g123_to_binary(g123, one_hots)

        # parse hidden
        new_episode = torch.all(hx.squeeze(0) == 0, dim=-1)
        hx = self.parse_hidden(hx)
        p = hx.p
        r = hx.r
        for x in hx:
            x.squeeze_(0)
        if torch.any(new_episode):
            p[new_episode, 0] = 1.  # initialize pointer to first subtask
            r[new_episode] = M[new_episode, 0]  # initialize r to first subtask
            # initialize g to first subtask
            hx.g[new_episode] = 0.

        def update_attention(p, t):
            p2 = F.pad(p, [1, 0])[:, :-1]
            p2[:, -1] += 1 - p2.sum(dim=-1)
            return p2

        return self.pack(
            self.inner_loop(
                a=hx.a,
                g=hx.g,
                M=M,
                M123=M123,
                N=N,
                T=T,
                float_subtask=hx.subtask,
                next_subtask=inputs.next_subtask,
                obs=obs,
                p=p,
                r=r,
                actions=actions,
                update_attention=update_attention))

    def pack(self, outputs):
        zipped = list(zip(*outputs))
        # for name, x in zip(RecurrentState._fields, zipped):
        # if not x:
        # print(name)
        # import ipdb
        # ipdb.set_trace()

        stacked = [torch.stack(x) for x in zipped]
        preprocessed = [x.float().view(*x.shape[:2], -1) for x in stacked]

        # for name, x, size in zip(RecurrentState._fields, preprocessed,
        # self.state_sizes):
        # if x.size(2) != size:
        # print(name, x, size)
        # import ipdb
        # ipdb.set_trace()
        # if x.dtype != torch.float32:
        # print(name)
        # import ipdb
        # ipdb.set_trace()

        hx = torch.cat(preprocessed, dim=-1)
        return hx, hx[-1]

    def inner_loop(self, a, g, M, M123, N, T, float_subtask, next_subtask, obs, p, r, actions,
                   update_attention):
        # combine past and present actions (sampled values)
        A = torch.cat([a.unsqueeze(0), actions.a], dim=0).long().squeeze(2)
        G = torch.cat([g.unsqueeze(0), actions.g], dim=0).long().squeeze(2)
        for t in range(T):
            subtask = float_subtask.long()
            float_subtask += next_subtask[t]

            agent_layer = obs[t, :, 6, :, :].long()
            j, k, l = torch.split(agent_layer.nonzero(), 1, dim=-1)

            def phi_update(subtask_param):
                debug_obs = obs[t, j, :, k, l].squeeze(1)
                task_sections = torch.split(subtask_param, tuple(self.subtask_nvec), dim=-1)
                parts = (debug_obs, self.a_one_hots[A[t]]) + task_sections
                if self.multiplicative_interaction:
                    return self.f(parts)
                outer_produce_obs = 1
                for i1, part in enumerate(parts):
                    for i2 in range(len(parts)):
                        if i1 != i2:
                            part.unsqueeze_(i2 + 1)
                    outer_produce_obs = outer_produce_obs * part

                c_logits = self.phi_update(outer_produce_obs.view(N, -1))
                if self.hard_update:
                    c_dist = FixedCategorical(logits=c_logits)
                    c = actions.c[t]
                    sample_new(c, c_dist)
                    probs = c_dist.probs
                    loss = -c_dist.log_probs(next_subtask[t])
                else:
                    c = torch.sigmoid(c_logits[:, :1])
                    probs = torch.zeros_like(c_logits)  # dummy value
                    loss = F.binary_cross_entropy(
                        torch.clamp(c, 0., 1.), next_subtask[t], reduction='none')
                return c, loss, probs

            # cr
            cr, cr_loss, cr_probs = phi_update(subtask_param=r, )

            # cg
            g_binary = M[torch.arange(N), G[t]]
            cg, cg_loss, cg_probs = phi_update(subtask_param=g_binary, )

            cr = cg = next_subtask[t]  # TODO

            # p
            p2 = update_attention(p, t)
            p = interp(p, p2, cr)

            # r
            r = (p.unsqueeze(1) @ M).squeeze(1)

            # g
            old_g = self.g_one_hots[G[t]]
            g_dist = FixedCategorical(probs=torch.clamp(interp(old_g, p, cg), 0., 1.))
            sample_new(G[t + 1], g_dist)

            # a
            g = G[t + 1]
            g_binary = M[torch.arange(N), g]
            conv_out = self.conv((obs[t], broadcast3d(g_binary, self.obs_shape[1:])))
            if self.agent is None:
                a_dist = self.actor(conv_out)
            else:
                agent_inputs = torch.cat(
                    ppo.subtasks.teacher.Obs(
                        base=obs[t].view(N, -1),
                        subtask=g.float().view(N, -1),
                        subtasks=M123.view(N, -1),
                    ),
                    dim=1)
                a_dist = self.agent(agent_inputs, rnn_hxs=None, masks=None).dist
            sample_new(A[t + 1], a_dist)
            # a[:] = 'wsadeq'.index(input('act:'))

            yield RecurrentState(
                cg=cg,
                cr=cr,
                cg_loss=cg_loss,
                cr_loss=cr_loss,
                cg_probs=cg_probs,
                cr_probs=cr_probs,
                p=p,
                r=r,
                g=G[t + 1],
                g_probs=g_dist.probs,
                g_loss=-g_dist.log_probs(subtask),
                a=A[t + 1],
                a_probs=a_dist.probs,
                subtask=float_subtask,
                v=self.critic(conv_out))
