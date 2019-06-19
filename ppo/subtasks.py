from collections import namedtuple

from gym.spaces import Box, Discrete
import numpy as np
import torch
from torch import nn as nn
import torch.jit
from torch.nn import functional as F

from ppo.agent import Agent, AgentValues, NNBase
from ppo.distributions import Categorical, DiagGaussian, FixedCategorical
from ppo.layers import Concat, Flatten, Parallel, Product, Reshape
from ppo.teacher import SubtasksTeacher
from ppo.utils import batch_conv1d, broadcast_3d, init_, interp, trace
from ppo.wrappers import SubtasksActions, get_subtasks_action_sections, get_subtasks_obs_sections

RecurrentState = namedtuple(
    'RecurrentState',
    'p r h b b_probs g_binary g_int g_probs c c_probs cg cg_loss cg_probs l l_probs a a_probs v c_truth '
    'c_loss l_loss g_loss b_loss subtask')


# noinspection PyMissingConstructor
class SubtasksAgent(Agent, NNBase):
    def __init__(self, obs_shape, action_space, task_space, hidden_size,
                 entropy_coef, alpha, zeta, hard_update, teacher_agent,
                 **kwargs):
        nn.Module.__init__(self)
        self.zeta = zeta
        self.alpha = alpha
        self.hard_update = hard_update
        self.entropy_coef = entropy_coef
        self.action_space = SubtasksActions(*action_space.spaces)
        self.teacher_agent = teacher_agent
        self.recurrent_module = SubtasksRecurrence(
            obs_shape=obs_shape,
            action_space=self.action_space,
            task_space=task_space,
            hidden_size=hidden_size,
            hard_update=hard_update,
            teacher_agent=teacher_agent,
            **kwargs,
        )
        self.obs_sections = get_subtasks_obs_sections(task_space)
        self.agent = teacher_agent

    def forward(self, inputs, rnn_hxs, masks, action=None,
                deterministic=False):
        obs, subtask, task, next_subtask = torch.split(
            inputs, self.obs_sections, dim=1)

        n = inputs.size(0)
        actions = None
        if action is not None:
            action_sections = get_subtasks_action_sections(self.action_space)
            actions = SubtasksActions(
                *torch.split(action, action_sections, dim=-1))

        all_hxs, last_hx = self._forward_gru(
            inputs.view(n, -1), rnn_hxs, masks, actions=actions)
        rm = self.recurrent_module
        hx = RecurrentState(*rm.parse_hidden(all_hxs))

        if action is None:
            actions = SubtasksActions(
                a=hx.a, b=hx.b, l=hx.l, c=hx.c, g_int=hx.g_int)

        if self.hard_update:
            dists = SubtasksActions(
                a=FixedCategorical(hx.a_probs),
                b=None,
                c=FixedCategorical(hx.c_probs),
                l=FixedCategorical(hx.l_probs),
                g_int=FixedCategorical(hx.g_probs),
            )
        else:
            dists = SubtasksActions(
                a=None if self.teacher_agent else FixedCategorical(hx.a_probs),
                b=None,
                c=None,
                l=None,
                g_int=FixedCategorical(hx.g_probs))

        log_probs = sum(
            dist.log_probs(a) for dist, a in zip(dists, actions)
            if dist is not None)
        entropies = sum(dist.entropy() for dist in dists if dist is not None)

        log = {k: v for k, v in hx._asdict().items() if k.endswith('_loss')}
        aux_loss = -self.entropy_coef * entropies.mean()

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
        all_hxs, last_hx = self._forward_gru(
            inputs.view(n, -1), rnn_hxs, masks)
        return self.recurrent_module.parse_hidden(all_hxs).v

    def _forward_gru(self, x, hxs, masks, actions=None):
        if actions is None:
            y = F.pad(x, [0, len(SubtasksActions._fields)], 'constant', -1)
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
    x[new] = dist.sample()[new].float()


class SubtasksRecurrence(torch.jit.ScriptModule):
    __constants__ = [
        'input_sections', 'subtask_space', 'state_sizes', 'recurrent'
    ]

    def __init__(self, obs_shape, action_space, task_space, hidden_size,
                 recurrent, hard_update, teacher_agent,
                 multiplicative_interaction):
        super().__init__()
        d, h, w = obs_shape
        conv_out_size = h * w * hidden_size
        subtask_space = list(map(int, task_space.nvec[0]))
        subtask_size = sum(subtask_space)
        n_subtasks = task_space.shape[0]
        self.multiplicative_interaction = multiplicative_interaction
        self.hard_update = hard_update
        self.obs_sections = get_subtasks_obs_sections(task_space)
        self.obs_shape = d, h, w
        self.task_nvec = task_space.nvec
        self.action_space = action_space
        self.n_subtasks = n_subtasks
        if teacher_agent:
            assert isinstance(teacher_agent, SubtasksTeacher)
        self.agent = teacher_agent

        # networks
        self.recurrent = recurrent

        self.conv1 = nn.Sequential(
            init_(
                nn.Conv2d(
                    self.obs_sections.base,
                    hidden_size,
                    kernel_size=3,
                    stride=1,
                    padding=1), 'relu'), nn.ReLU(), Flatten())

        self.conv = nn.Sequential(
            Concat(dim=1),
            init_(
                nn.Conv2d(
                    self.obs_sections.base + self.obs_sections.subtask,
                    hidden_size,
                    kernel_size=3,
                    stride=1,
                    padding=1), 'relu'), nn.ReLU(), Flatten())

        input_size = h * w * hidden_size  # conv output
        if isinstance(action_space.a, Discrete):
            num_outputs = action_space.a.n
            self.actor = Categorical(input_size, num_outputs)
        elif isinstance(action_space.a, Box):
            num_outputs = action_space.a.shape[0]
            self.actor = DiagGaussian(input_size, num_outputs)
        else:
            raise NotImplementedError

        self.critic = init_(nn.Linear(input_size, 1))

        self.f = nn.Sequential(
            Parallel(
                init_(nn.Linear(self.obs_sections.base, hidden_size)),
                init_(nn.Linear(action_space.a.n, hidden_size)),
                *[init_(nn.Linear(i, hidden_size)) for i in self.task_nvec[0]],
            ),
            Product(),
        )

        subcontroller = nn.GRUCell if recurrent else nn.Linear
        self.subcontroller = trace(
            lambda in_size: nn.Sequential(
                init_(subcontroller(in_size, hidden_size), 'relu'),
                nn.ReLU(),
            ),
            in_size=conv_out_size)  # h

        self.phi_update = trace(
            lambda in_size: init_(nn.Linear(in_size, 1), 'sigmoid'),
            in_size=self.obs_sections.base * action_space.a.n * int(
                task_space.nvec[0].prod()))

        self.phi_shift = trace(
            lambda in_size: nn.Sequential(
                init_(nn.Linear(in_size, 3)),  # 3 for {-1, 0, +1}
            ),
            in_size=hidden_size)

        self.pi_theta = Categorical(subtask_size, action_space.g_int.n)

        self.beta = Categorical(
            conv_out_size +  # x
            subtask_size,  # g
            2)

        for i, d in enumerate(self.task_nvec[0]):
            self.register_buffer(f'part{i}_one_hot', torch.eye(int(d)))
        self.register_buffer('l_one_hots', torch.eye(3))
        self.register_buffer('p_one_hots', torch.eye(self.n_subtasks))
        self.register_buffer('a_one_hots', torch.eye(int(action_space.a.n)))
        self.register_buffer('g_one_hots', torch.eye(action_space.g_int.n))
        self.register_buffer('subtask_space',
                             torch.tensor(task_space.nvec[0].astype(np.int64)))

        state_sizes = RecurrentState(
            p=self.n_subtasks,
            r=subtask_size,
            h=hidden_size,
            g_binary=subtask_size,
            g_int=1,
            b=1,
            b_probs=2,
            g_probs=self.n_subtasks,
            c=1,
            c_truth=1,
            c_probs=2,
            cg=1,
            cg_loss=1,
            cg_probs=2,
            l=1,
            a=1,
            v=1,
            a_probs=action_space.a.n,
            l_probs=3,
            c_loss=1,
            l_loss=1,
            g_loss=1,
            b_loss=1,
            subtask=1)
        self.state_sizes = RecurrentState(*map(int, state_sizes))

    # @torch.jit.script_method
    def parse_hidden(self, hx):
        return RecurrentState(*torch.split(hx, self.state_sizes, dim=-1))

    # @torch.jit.script_method
    def g123_to_binary(self, *g123):
        return torch.cat([
            getattr(self, f'part{i}_one_hot')[g.long()]
            for i, g in enumerate(g123)
        ],
                         dim=-1)

    # @torch.jit.script_method
    def g_binary_to_int(self, g_binary):
        g123 = g_binary.nonzero()[:, 1:].view(-1, 3)
        g123 -= F.pad(
            torch.cumsum(self.subtask_space, dim=0)[:2], [1, 0], 'constant', 0)
        g123[:, :-1] *= self.subtask_space[1:]  # g1 * x2, g2 * x3
        g123[:, 0] *= self.subtask_space[2]  # g1 * x3
        return g123.sum(dim=-1)

    # @torch.jit.script_method
    def g_int_to_123(self, g):
        x1, x2, x3 = self.subtask_space.to(g.dtype)
        g1 = g // (x2 * x3)
        x4 = g % (x2 * x3)
        g2 = x4 // x3
        g3 = x4 % x3
        return g1, g2, g3

    def g_int_to_binary(self, g):
        return self.g123_to_binary(*self.g_int_to_123(g)).squeeze(1)

    def check_grad(self, **kwargs):
        for k, v in kwargs.items():
            if v.grad_fn is not None:
                grads = torch.autograd.grad(
                    v.mean(),
                    self.parameters(),
                    retain_graph=True,
                    allow_unused=True)
                for (name, _), grad in zip(self.named_parameters(), grads):
                    if grad is None:
                        print(f'{k} has no grad wrt {name}')
                    else:
                        print(
                            f'mean grad ({v.mean().item()}) of {k} wrt {name}:',
                            grad.mean())
                        if torch.isnan(grad.mean()):
                            import ipdb
                            ipdb.set_trace()

    # @torch.jit.script_method
    def forward(self, inputs, hx):
        assert hx is not None
        T, N, D = inputs.shape
        inputs = inputs.view(T, N, -1)
        n_actions = len(SubtasksActions._fields)
        inputs, *actions = torch.split(
            inputs.detach(), [D - n_actions] + [1] * n_actions, dim=2)
        actions = SubtasksActions(*actions)
        inputs = inputs.view(T, N, *self.obs_shape)
        obs, subtasks, task_broad, next_subtask_broad = torch.split(
            inputs, self.obs_sections, dim=2)
        task = task_broad[:, :, :, 0, 0]
        next_subtask = next_subtask_broad[:, :, :, 0, 0]
        sections = [self.n_subtasks] * self.task_nvec.shape[1]
        task_type, count, obj = torch.split(task, sections, dim=-1)

        M = self.g123_to_binary(task_type[0], (count - 1)[0], obj[0])
        new_episode = torch.all(hx.squeeze(0) == 0, dim=-1)
        hx = self.parse_hidden(hx)

        h = hx.h
        p = hx.p
        r = hx.r
        g_binary = hx.g_binary
        float_subtask = hx.subtask
        for x in hx:
            x.squeeze_(0)

        if torch.any(new_episode):
            p[new_episode, 0] = 1.  # initialize pointer to first subtask
            r[new_episode] = M[new_episode, 0]  # initialize r to first subtask
            g0 = M[new_episode, 0]
            g_binary[new_episode] = g0  # initialize g_binary to first subtask
            hx.g_int[new_episode] = 0.

        a_ints = torch.cat([hx.a.unsqueeze(0), actions.a], dim=0)
        g_ints = torch.cat([hx.g_int.unsqueeze(0), actions.g_int], dim=0)

        outputs = RecurrentState(*[[] for _ in RecurrentState._fields])

        n = obs.size(0)
        for i in range(n):
            subtask = float_subtask.long()
            float_subtask += next_subtask[i]
            outputs.subtask.append(float_subtask)

            conv_out = self.conv1(obs[i])

            idxs = a_ints[i].flatten().long()
            agent_layer = obs[i, :, 6, :, :].long()

            def phi_update(subtask_param, values, losses, probs):
                j, k, l = torch.split(agent_layer.nonzero(), [1, 1, 1], dim=-1)
                debug_obs = obs[i, j, :, k, l].squeeze(1)
                task_sections = torch.split(
                    subtask_param, tuple(self.task_nvec[0]), dim=-1)
                parts = (debug_obs, self.a_one_hots[idxs]) + task_sections
                if self.multiplicative_interaction:
                    return self.f(parts)
                obs4d = 1
                for i1, part in enumerate(parts):
                    for i2 in range(len(parts)):
                        if i1 != i2:
                            part.unsqueeze_(i2 + 1)
                    obs4d = obs4d * part

                logits = self.phi_update(obs4d.view(N, -1))
                if self.hard_update:
                    dist = FixedCategorical(logits=logits)
                    new = actions.c[i] < 0
                    c = actions.c[i].clone()
                    c[new] = dist.sample()[new].float()
                    probs.append(dist.probs)
                    losses.append(-dist.log_probs(next_subtask[i]))
                else:
                    c = torch.sigmoid(logits[:, :1])
                    probs.append(torch.zeros((N, 2),
                                             device=c.device))  # dummy value
                    if torch.any(next_subtask[i] > 0):
                        weight = torch.ones_like(c)
                        weight[next_subtask[i] > 0] /= torch.sum(
                            next_subtask[i] > 0)
                        weight[next_subtask[i] == 0] /= torch.sum(
                            next_subtask[i] == 0)

                        losses.append(
                            F.binary_cross_entropy(
                                torch.clamp(c, 0., 1.),
                                next_subtask[i],
                                weight=weight,
                                reduction='none'))
                    else:
                        losses.append(torch.zeros_like(c))
                values.append(c)
                return c

            c = phi_update(
                subtask_param=hx.r,
                values=outputs.c,
                losses=outputs.c_loss,
                probs=outputs.c_probs)

            outputs.c_truth.append(next_subtask[i])
            # TODO: figure this out
            # if self.recurrent:
            #     h2 = self.subcontroller(obs[i], h)
            # else:
            # h2 = self.subcontroller(conv_out)

            logits = self.phi_shift(h)
            l_target = 1 - next_subtask[i].long().flatten()
            if self.hard_update:
                dist = FixedCategorical(logits=logits)
                sample_new(actions.l[i], dist)
                outputs.l.append(actions.l[i].float())
                outputs.l_probs.append(dist.probs)
                l = self.l_one_hots[actions.l[i].long().flatten()]
                outputs.l_loss.append(-dist.log_probs(l_target))
            else:
                l = F.softmax(logits, dim=1)
                outputs.l.append(torch.zeros_like(l)[:, :1])  # dummy value
                outputs.l_probs.append(torch.zeros_like(l))  # dummy value
                outputs.l_loss.append(
                    F.cross_entropy(
                        logits,
                        l_target,
                        reduction='none',
                    ).unsqueeze(1))
                l_idxs = torch.zeros_like(c).long().flatten()
                l = self.l_one_hots[l_idxs]  # TODO

            p2 = batch_conv1d(p, l)
            p2[:, -1] += 1 - p2.sum(dim=-1)
            p2 = torch.clamp(p2, 0., 1.)
            r2 = p2 @ M
            p = interp(p, p2.squeeze(1), c)
            r = interp(r, r2.squeeze(1), c)
            outputs.p.append(p)
            outputs.r.append(r)
            outputs.h.append(h)

            # TODO: deterministic
            # r_repl
            r_repl = []
            for j in range(N):
                r_repl.append(M[j, subtask[j]])
            r_repl = torch.stack(r_repl).squeeze(1).detach()

            # g
            cg = phi_update(
                subtask_param=hx.g_binary,
                values=outputs.cg,
                losses=outputs.cg_loss,
                probs=outputs.cg_probs)
            old_g = self.g_one_hots[g_ints[i].long().flatten()]
            dist = FixedCategorical(
                probs=torch.clamp(interp(old_g, p, cg), 0., 1.))
            # dist = FixedCategorical(probs=p)
            sample_new(g_ints[i + 1], dist)
            outputs.g_int.append(g_ints[i + 1])
            outputs.g_probs.append(dist.probs)

            # g_loss
            # g_target = []
            # for j in range(N):
            # g_target.append(self.g_binary_to_int(M[j, subtask[j]]))
            # g_target = torch.stack(g_target).detach()
            outputs.g_loss.append(-dist.log_probs(subtask))

            g_idxs = g_ints[i + 1].long().flatten()
            g_binary = M[torch.arange(N), g_idxs]
            outputs.g_binary.append(g_binary)

            # b
            dist = self.beta(torch.cat([conv_out, g_binary], dim=-1))
            sample_new(actions.b[i], dist)
            outputs.b_probs.append(dist.probs)

            # b_loss
            outputs.b_loss.append(-dist.log_probs(next_subtask[i]))
            outputs.b.append(actions.b[i])

            # a
            g_broad = broadcast_3d(g_binary, self.obs_shape[1:])
            conv_out2 = self.conv((obs[i], g_broad))
            if self.agent is None:
                dist = self.actor(conv_out2)
            else:
                g = broadcast_3d(g_binary, obs.shape[3:])
                teacher_inputs = torch.cat(
                    [obs[i], g, task_broad[i], next_subtask_broad[i]], dim=1)
                dist = self.agent(
                    teacher_inputs, rnn_hxs=None, masks=None).dist
            sample_new(a_ints[i + 1], dist)
            # a[:] = 'wsadeq'.index(input('act:'))

            outputs.a.append(a_ints[i + 1])
            outputs.a_probs.append(dist.probs)

            # v
            outputs.v.append(self.critic(conv_out2))

        stacked = []

        # for name, x in zip(RecurrentState._fields, outputs):
        #     if not x:
        #         print(name)
        #         import ipdb
        #         ipdb.set_trace()

        for x in outputs:
            stacked.append(torch.stack(x))

        # for name, x, size in zip(RecurrentState._fields, stacked,
        #                          self.state_sizes):
        #     if x.size(2) != size:
        #         print(name, x, size)
        #         import ipdb
        #         ipdb.set_trace()
        #     if x.dtype != torch.float32:
        #         print(name)
        #         import ipdb
        #         ipdb.set_trace()

        hx = torch.cat(stacked, dim=-1)
        return hx, hx[-1]
