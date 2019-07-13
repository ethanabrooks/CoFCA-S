from collections import namedtuple
import itertools

from gym.spaces import Box, Discrete, MultiDiscrete
import numpy as np
import torch
from torch import nn as nn
import torch.jit
from torch.nn import functional as F

from gridworld_env.subtasks_gridworld import Obs
import ppo
from ppo.agent import AgentValues, NNBase
from ppo.distributions import Categorical, DiagGaussian, FixedCategorical
from ppo.layers import Concat, Flatten, Parallel, Product, Reshape, ShallowCopy, Sum
import ppo.subtasks.teacher
from ppo.subtasks.teacher import Teacher, g_binary_to_discrete, g_discrete_to_binary
from ppo.subtasks.wrappers import Actions
from ppo.utils import broadcast3d, init_, interp, trace

RecurrentState = namedtuple(
    "RecurrentState", "a cg cr r p g a_probs cg_probs cr_probs g_probs v g_loss subtask"
)


# noinspection PyMissingConstructor
class Agent(ppo.agent.Agent, NNBase):
    def __init__(
        self,
        obs_space,
        action_space,
        hidden_size,
        entropy_coef,
        hard_update,
        agent_load_path,
        agent_args,
        **kwargs,
    ):
        nn.Module.__init__(self)
        self.hard_update = hard_update
        self.entropy_coef = entropy_coef
        self.action_spaces = Actions(**action_space.spaces)
        self.obs_space = obs_space
        obs_spaces = Obs(**self.obs_space.spaces)
        agent = None
        if agent_load_path is not None:
            agent_obs_spaces = ppo.subtasks.teacher.Obs(
                base=obs_spaces.base, subtask=MultiDiscrete(obs_spaces.subtasks.nvec[0])
            )
            agent = self.load_agent(
                agent_load_path=agent_load_path,
                obs_spaces=agent_obs_spaces,
                action_spaces=self.action_spaces,
                **agent_args,
            )
        self.recurrent_module = self.build_recurrent_module(
            agent=agent,
            hard_update=hard_update,
            hidden_size=hidden_size,
            obs_spaces=obs_spaces,
            action_spaces=self.action_spaces,
            **kwargs,
        )

    def load_agent(self, agent_load_path, device, **agent_args):
        agent = ppo.subtasks.Teacher(**agent_args)

        state_dict = torch.load(agent_load_path, map_location=device)
        assert "vec_normalize" not in state_dict, "oy"
        state_dict["agent"].update(
            part0_one_hot=agent.part0_one_hot,
            part1_one_hot=agent.part1_one_hot,
            part2_one_hot=agent.part2_one_hot,
        )
        agent.load_state_dict(state_dict["agent"])
        print(f"Loaded teacher parameters from {agent_load_path}.")
        return agent

    def build_recurrent_module(self, **kwargs):
        return Recurrence(**kwargs)

    def forward(self, inputs, rnn_hxs, masks, action=None, deterministic=False):
        n = inputs.size(0)
        actions = None
        if action is not None:
            actions = Actions(
                *torch.split(action, [1] * len(self.action_spaces), dim=-1)
            )

        all_hxs, last_hx = self._forward_gru(
            inputs.view(n, -1), rnn_hxs, masks, actions=actions
        )
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
                a=None
                if rm.agent  # use pre-trained agent so don't train
                else FixedCategorical(hx.a_probs),
                cg=None,
                cr=None,
                g=FixedCategorical(hx.g_probs),
            )

        log_probs = sum(
            dist.log_probs(a) for dist, a in zip(dists, actions) if dist is not None
        )
        entropies = sum(dist.entropy() for dist in dists if dist is not None)
        aux_loss = -self.entropy_coef * entropies.mean()
        log = {k: v for k, v in hx._asdict().items() if k.endswith("_loss")}

        return AgentValues(
            value=hx.v,
            action=torch.cat(actions, dim=-1),
            action_log_probs=log_probs,
            aux_loss=aux_loss,
            rnn_hxs=torch.cat(hx, dim=-1),
            dist=None,
            log=log,
        )

    def get_value(self, inputs, rnn_hxs, masks):
        n = inputs.size(0)
        all_hxs, last_hx = self._forward_gru(inputs.view(n, -1), rnn_hxs, masks)
        return self.recurrent_module.parse_hidden(all_hxs).v

    def _forward_gru(self, x, hxs, masks, actions=None):
        if actions is None:
            y = F.pad(x, [0, len(Actions._fields)], "constant", -1)
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

        self.phi_update = trace(
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
        )
        self.state_sizes = RecurrentState(*map(int, state_sizes))

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

    def parse_inputs(self, inputs):
        return Obs(*torch.split(inputs, self.obs_sections, dim=2))

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

        def update_attention(p, t):
            p2 = F.pad(p, [1, 0])[:, :-1]
            p2[:, -1] += 1 - p2.sum(dim=-1)
            return p2

        def gating_function(subtask_param, obs, action):
            task_sections = torch.split(subtask_param, tuple(self.subtask_nvec), dim=-1)
            # NOTE {
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
            parts = (self.conv1(obs), self.a_one_hots[action]) + task_sections
            outer_product_obs = 1
            for i1, part in enumerate(parts):
                for i2 in range(len(parts)):
                    if i1 != i2:
                        part.unsqueeze_(i2 + 1)
                outer_product_obs = outer_product_obs * part

            c_logits = self.phi_update(outer_product_obs.view(N, -1))
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

        return self.pack(
            self.inner_loop(
                inputs=inputs,
                a=hx.a,
                g=hx.g,
                cr=hx.cr,
                cg=hx.cg,
                M=M,
                M_discrete=M_discrete,
                N=N,
                T=T,
                subtask=inputs.subtask,
                p=p,
                r=r,
                actions=actions,
                update_attention=update_attention,
                gating_function=gating_function,
            )
        )

    @property
    def g_discrete_one_hots(self):
        for i in itertools.count():
            try:
                yield getattr(self, f"part{i}_one_hot")
            except AttributeError:
                break

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

    def get_a_dist(self, conv_out, g_binary, obs):
        N = obs.size(0)
        if self.agent is None:
            return self.actor(conv_out)
        else:
            agent_inputs = ppo.subtasks.teacher.Obs(
                base=obs.view(N, -1), subtask=g_binary
            )
            return self.agent(agent_inputs, rnn_hxs=None, masks=None).dist

    def inner_loop(
        self,
        g,
        a,
        cr,
        cg,
        M,
        M_discrete,
        N,
        T,
        subtask,
        p,
        r,
        actions,
        update_attention,
        gating_function,
        inputs,
    ):
        subtask = subtask.long().view(T, N)
        # combine past and present actions (sampled values)
        obs = inputs.base
        A = torch.cat([actions.a, a.unsqueeze(0)], dim=0).long().squeeze(2)
        G = torch.cat([actions.g, g.unsqueeze(0)], dim=0).long().squeeze(2)
        for t in range(T):

            # agent_layer = obs[t, :, 6, :, :].long()
            # j, k, l = torch.split(agent_layer.nonzero(), 1, dim=-1)

            # p
            p2 = update_attention(p, t)
            p = interp(p, p2, cr)

            # r
            r = (p.unsqueeze(1) @ M).squeeze(1)

            # g
            old_g = self.g_one_hots[G[t - 1]]
            probs = interp(old_g, p, cg)
            g_dist = FixedCategorical(probs=torch.clamp(probs, 0.0, 1.0))
            sample_new(G[t], g_dist)

            # a
            g = G[t]
            g_binary = M[torch.arange(N), g]
            conv_out = self.conv2((obs[t], broadcast3d(g_binary, self.obs_shape[1:])))
            a_dist = self.get_a_dist(conv_out, g_binary, obs[t])
            sample_new(A[t], a_dist)
            # a[:] = 'wsadeq'.index(input('act:'))

            # cr
            cr, cr_probs = gating_function(subtask_param=r, obs=obs[t], action=A[t])

            # cg
            g = M[torch.arange(N), G[t]]
            cg, cg_probs = gating_function(subtask_param=g, obs=obs[t], action=A[t])

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
            )
