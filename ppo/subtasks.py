from collections import namedtuple

import numpy as np
import torch
import torch.jit
from gym import spaces
from gym.spaces import Box, Discrete
from torch import nn as nn
from torch.nn import functional as F

from ppo.agent import Agent, AgentValues, NNBase
from ppo.distributions import Categorical, DiagGaussian, FixedCategorical
from ppo.layers import Broadcast3d, Concat, Flatten, Reshape
from ppo.teacher import SubtasksTeacher
from ppo.utils import batch_conv1d, broadcast_3d, init_, interp, trace
from ppo.wrappers import (SubtasksActions, get_subtasks_action_sections,
                          get_subtasks_obs_sections)

RecurrentState = namedtuple(
    'RecurrentState',
    'p r h b b_probs g g_int g_probs c c_probs l l_probs a a_probs v c_guess '
    'c_loss l_loss p_loss r_loss g_loss b_loss subtask')


# noinspection PyMissingConstructor
class SubtasksAgent(Agent, NNBase):
    def __init__(self,
                 obs_shape,
                 action_space,
                 task_space,
                 hidden_size,
                 entropy_coef,
                 alpha,
                 zeta,
                 hard_update,
                 teacher_agent=None,
                 **kwargs):
        nn.Module.__init__(self)
        self.zeta = zeta
        self.alpha = alpha
        self.hard_update = hard_update
        if teacher_agent:
            assert isinstance(teacher_agent, SubtasksTeacher)
        self.teacher_agent = teacher_agent
        self.entropy_coef = entropy_coef
        self.action_space = SubtasksActions(*action_space.spaces)
        self.recurrent_module = SubtasksRecurrence(
            obs_shape=obs_shape,
            action_space=self.action_space,
            task_space=task_space,
            hidden_size=hidden_size,
            hard_update=hard_update,
            **kwargs,
        )
        self.obs_sections = get_subtasks_obs_sections(task_space)

    def forward(self, inputs, rnn_hxs, masks, action=None,
                deterministic=False):
        obs, g_target, task, next_subtask = torch.split(
            inputs, self.obs_sections, dim=1)

        n = inputs.shape[0]
        all_hxs, last_hx = self._forward_gru(
            inputs.view(n, -1), rnn_hxs, masks)
        hx = RecurrentState(*self.recurrent_module.parse_hidden(all_hxs))

        # print('g       ', hx.g[0])
        # print('g_target', g_target[0, :, 0, 0])

        if self.hard_update:
            dists = SubtasksActions(
                a=FixedCategorical(hx.a_probs),
                b=FixedCategorical(hx.b_probs),
                c=FixedCategorical(hx.c_probs),
                g=FixedCategorical(hx.g_probs),
                l=FixedCategorical(hx.l_probs),
                g_int=None)
        else:
            dists = SubtasksActions(
                a=FixedCategorical(hx.a_probs),
                b=FixedCategorical(hx.b_probs),
                c=None,
                g=FixedCategorical(hx.g_probs),
                l=None,
                g_int=None)

        if action is None:
            actions = SubtasksActions(
                a=hx.a, b=hx.b, g=hx.g, l=hx.l, c=hx.c, g_int=hx.g_int)
        else:
            action_sections = get_subtasks_action_sections(self.action_space)
            actions = SubtasksActions(
                *torch.split(action, action_sections, dim=-1))

        log_probs1 = dists.a.log_probs(actions.a) + dists.b.log_probs(
            actions.b)
        log_probs2 = dists.g.log_probs(actions.g_int)
        entropies1 = dists.a.entropy() + dists.b.entropy()
        entropies2 = dists.g.entropy()
        if self.hard_update:
            log_probs1 += dists.c.log_probs(actions.c)
            # log_probs2 += dists.l.log_probs(actions.l)
            entropies1 += dists.c.entropy()
            # entropies2 += dists.l.entropy()

        g_accuracy = torch.all(hx.g.round() == g_target[:, :, 0, 0], dim=-1)

        c_accuracy = torch.mean((hx.c_guess.round() == hx.c).float())
        c_precision = torch.mean(
            (hx.c_guess.round()[hx.c_guess > 0] == hx.c[hx.c_guess > 0]
             ).float())
        c_recall = torch.mean(
            (hx.c_guess.round()[hx.c > 0] == hx.c[hx.c > 0]).float())
        log = dict(
            g_accuracy=g_accuracy.float(),
            c_accuracy=c_accuracy,
            c_recall=c_recall,
            c_precision=c_precision)
        aux_loss = self.alpha * hx.c_loss - self.entropy_coef * (
            entropies1 + entropies2)

        if self.teacher_agent:
            imitation_dist = self.teacher_agent(inputs, rnn_hxs, masks).dist
            imitation_probs = imitation_dist.probs.detach().unsqueeze(1)
            our_log_probs = torch.log(dists.a.probs).unsqueeze(2)
            imitation_obj = (imitation_probs @ our_log_probs).view(-1)
            log.update(imitation_obj=imitation_obj)
            aux_loss -= imitation_obj

        log_probs = log_probs1 + hx.c * log_probs2

        g_broad = broadcast_3d(actions.g, obs.shape[2:])
        value = self.recurrent_module.critic(
            self.recurrent_module.conv2((obs, g_broad)))

        for k, v in hx._asdict().items():
            if k.endswith('_loss'):
                log[k] = v

        return AgentValues(
            value=value,
            action=torch.cat(actions, dim=-1),
            action_log_probs=log_probs,
            aux_loss=aux_loss.mean(),
            rnn_hxs=torch.cat(hx, dim=-1),
            dist=None,
            log=log)

    def get_value(self, inputs, rnn_hxs, masks):
        n = inputs.shape[0]
        all_hxs, last_hx = self._forward_gru(
            inputs.view(n, -1), rnn_hxs, masks)
        return self.recurrent_module.parse_hidden(all_hxs).v

    @property
    def recurrent_hidden_state_size(self):
        return sum(self.recurrent_module.state_sizes)

    @property
    def is_recurrent(self):
        return True


class SubtasksRecurrence(torch.jit.ScriptModule):
    __constants__ = [
        'input_sections', 'subtask_space', 'state_sizes', 'recurrent'
    ]

    def __init__(self, obs_shape, action_space, task_space, hidden_size,
                 recurrent, hard_update, multiplicative_interaction):
        super().__init__()
        d, h, w = obs_shape
        conv_out_size = h * w * hidden_size
        self.subtask_space = list(map(int, task_space.nvec[0]))
        self.hard_update = hard_update
        subtask_size = sum(self.subtask_space)
        n_subtasks = task_space.shape[0]
        self.obs_sections = get_subtasks_obs_sections(task_space)
        self.obs_shape = d, h, w

        # networks
        self.recurrent = recurrent

        self.debug = nn.Sequential(
            init_(
                nn.Linear(
                    1 + int(task_space.nvec[0].sum()) *
                    (self.obs_sections.base + action_space.a.n), 1),
                'sigmoid'), )

        self.conv1 = nn.Sequential(
            init_(
                nn.Conv2d(
                    self.obs_sections.base,
                    hidden_size,
                    kernel_size=3,
                    stride=1,
                    padding=1), 'relu'), nn.ReLU(), Flatten())

        if multiplicative_interaction:
            conv_weight_shape = hidden_size, self.obs_sections.base, 3, 3
            self.conv_weight = nn.Sequential(
                nn.Linear(self.obs_sections.subtask,
                          np.prod(conv_weight_shape)),
                Reshape(-1, *conv_weight_shape))

        else:
            self.conv2 = nn.Sequential(
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

        in_size = (
            conv_out_size +  # x
            subtask_size +  # r
            subtask_size +  # g
            1)  # b
        self.f = nn.Sequential(
            init_(nn.Linear(in_size, hidden_size), 'relu'),
            nn.ReLU(),
        )

        subcontroller = nn.GRUCell if recurrent else nn.Linear
        self.subcontroller = trace(
            lambda in_size: nn.Sequential(
                init_(subcontroller(in_size, hidden_size), 'relu'),
                nn.ReLU(),
            ),
            in_size=conv_out_size)  # h

        self.phi_update = trace(
            lambda in_size: init_(nn.Linear(in_size, 2), 'sigmoid'),
            in_size=(
                hidden_size +  # s
                hidden_size))  # h

        self.phi_shift = trace(
            lambda in_size: nn.Sequential(
                init_(nn.Linear(in_size, hidden_size), 'relu'),
                nn.ReLU(),
                init_(nn.Linear(hidden_size, 3)),  # 3 for {-1, 0, +1}
            ),
            in_size=hidden_size)

        self.pi_theta = nn.Sequential(
            Concat(dim=-1),
            Broadcast3d(h, w),
            torch.jit.trace(
                nn.Sequential(
                    init_(
                        nn.Conv2d(
                            (
                                subtask_size +  # r
                                hidden_size),  # h
                            hidden_size,
                            kernel_size=3,
                            stride=1,
                            padding=1),
                        'relu'),
                    nn.ReLU(),
                    Flatten(),
                ),
                example_inputs=torch.rand(1, subtask_size + hidden_size, h, w),
            ),
            Categorical(h * w * hidden_size, np.prod(self.subtask_space)),
        )

        self.beta = Categorical(
            conv_out_size +  # x
            subtask_size,  # g
            2)

        # embeddings
        for name, d in zip(
            ['type_embeddings', 'count_embeddings', 'obj_embeddings'],
                self.subtask_space):
            self.register_buffer(name, torch.eye(int(d)))

        self.register_buffer('l_targets', torch.tensor([[1], [2]]))
        self.register_buffer('l_values', torch.eye(3))
        self.register_buffer('p_values', torch.eye(n_subtasks))
        self.register_buffer('a_values', torch.eye(action_space.a.n))

        self.task_sections = [n_subtasks] * task_space.nvec.shape[1]
        state_sizes = RecurrentState(
            p=n_subtasks,
            r=subtask_size,
            h=hidden_size,
            g=subtask_size,
            g_int=1,
            b=1,
            b_probs=2,
            g_probs=np.prod(self.subtask_space),
            c=1,
            c_guess=1,
            c_probs=2,
            l=1,
            a=1,
            v=1,
            a_probs=action_space.a.n,
            l_probs=3,
            c_loss=1,
            l_loss=1,
            p_loss=1,
            r_loss=1,
            g_loss=1,
            b_loss=1,
            subtask=1)
        self.state_sizes = RecurrentState(*map(int, state_sizes))

    # @torch.jit.script_method
    def parse_hidden(self, hx):
        return RecurrentState(*torch.split(hx, self.state_sizes, dim=-1))

    @torch.jit.script_method
    def embed_task(self, task_type, count, obj):
        return torch.cat([
            self.type_embeddings[task_type.long()],
            self.count_embeddings[count.long()],
            self.obj_embeddings[obj.long()],
        ],
                         dim=-1)

    def encode(self, g1, g2, g3):
        x1, x2, x3 = self.subtask_space
        return (g1 * (x2 * x3) + g2 * x3 + g3).long()

    def decode(self, g):
        x1, x2, x3 = self.subtask_space
        g1 = g // (x2 * x3)
        x4 = g % (x2 * x3)
        g2 = x4 // x3
        g3 = x4 % x3
        return g1, g2, g3

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
        T, N, _ = inputs.shape
        inputs = inputs.view(T, N, *self.obs_shape)

        obs, subtasks, task, next_subtask = torch.split(
            inputs, self.obs_sections, dim=2)
        subtasks = subtasks[:, :, :, 0, 0]
        task = task[:, :, :, 0, 0]
        next_subtask = next_subtask[:, :, :, 0, 0]
        task_type, count, obj = torch.split(task, self.task_sections, dim=-1)

        M = self.embed_task(task_type[0], (count - 1)[0], obj[0])
        new_episode = torch.all(hx.squeeze(0) == 0, dim=-1)
        hx = self.parse_hidden(hx)

        p = hx.p
        r = hx.r
        g = hx.g
        b = hx.b
        h = hx.h
        float_subtask = hx.subtask

        for x in hx:
            x.squeeze_(0)

        p[new_episode, 0] = 1.  # initialize pointer to first subtask
        r[new_episode] = M[new_episode, 0]  # initialize r to first subtask
        g[new_episode] = M[new_episode, 0]  # initialize g to first subtask

        outputs = RecurrentState(*[[] for _ in RecurrentState._fields])

        n = obs.shape[0]
        for i in range(n):
            float_subtask += next_subtask[i]
            outputs.subtask.append(float_subtask)
            subtask = float_subtask.long()
            m = M.shape[0]
            conv_out = self.conv1(obs[i])

            s = self.f(torch.cat([conv_out, r, g, b], dim=-1))
            logits = self.phi_update(torch.cat([s, h], dim=-1))
            if self.hard_update:
                dist = FixedCategorical(logits=logits)
                c = dist.sample().float()
                outputs.c_probs.append(dist.probs)
            else:
                c = torch.sigmoid(logits[:, :1])
                outputs.c_probs.append(torch.zeros_like(logits))  # dummy value

            a_idxs = hx.a.flatten().long()
            agent_layer = obs[i, :, 6, :, :].long()
            j, k, l = torch.split(agent_layer.nonzero(), [1, 1, 1], dim=-1)
            debug_obs = obs[i, j, :, k, l].squeeze(1)
            part1 = subtasks[i].unsqueeze(1) * debug_obs.unsqueeze(2)
            part2 = subtasks[i].unsqueeze(1) * self.a_values[a_idxs].unsqueeze(
                2)
            cat = torch.cat([part1, part2], dim=1)
            bsize = cat.shape[0]
            reshape = cat.view(bsize, -1)
            debug_in = torch.cat([reshape, next_subtask[i]], dim=-1)

            # print(debug_in[:, [39, 30, 21, 12, 98, 89]])
            # print(next_subtask[i])
            c = torch.sigmoid(self.debug(debug_in))
            outputs.c_guess.append(c)

            if torch.any(next_subtask[i] > 0):
                weight = torch.ones_like(c)
                weight[next_subtask[i] > 0] /= torch.sum(next_subtask[i] > 0)
                weight[next_subtask[i] == 0] /= torch.sum(next_subtask[i] == 0)
            else:
                weight = None

            outputs.c_loss.append(
                F.binary_cross_entropy(
                    torch.clamp(c, 0., 1.),
                    next_subtask[i],
                    weight=weight,
                    reduction='none'))

            c = next_subtask[i]  # TODO
            outputs.c.append(c)

            # TODO: figure this out
            # if self.recurrent:
            #     h2 = self.subcontroller(obs[i], h)
            # else:
            h2 = self.subcontroller(conv_out)

            logits = self.phi_shift(h2)
            # if self.hard_update:
            # dist = FixedCategorical(logits=logits)
            # l = dist.sample()
            # outputs.l.append(l.float())
            # outputs.l_probs.append(dist.probs)
            # l = self.l_values[l]
            # else:
            l = F.softmax(logits, dim=1)
            outputs.l.append(torch.zeros_like(c))  # dummy value
            outputs.l_probs.append(torch.zeros_like(l))  # dummy value

            # l_loss
            l_target = self.l_targets[next_subtask[i].long()].view(-1)
            outputs.l_loss.append(
                F.cross_entropy(
                    logits,
                    l_target,
                    reduction='none',
                ).unsqueeze(1))

            p2 = batch_conv1d(p, l)

            # p_losss
            outputs.p_loss.append(
                F.cross_entropy(
                    p2.squeeze(1), subtask.squeeze(1),
                    reduction='none').unsqueeze(1))

            r2 = p2 @ M

            # r_loss
            r_target = []
            for j in range(m):
                r_target.append(M[j, subtask[j]])
            r_target = torch.cat(r_target).detach()
            r_loss = F.binary_cross_entropy(
                torch.clamp(r2.squeeze(1), 0., 1.),
                r_target,
                reduction='none',
            )
            outputs.r_loss.append(torch.mean(r_loss, dim=-1, keepdim=True))

            p = interp(p, p2, c)
            r = interp(r, r2, c)
            h = interp(h, h2, c)

            outputs.p.append(p)
            outputs.r.append(r)
            outputs.h.append(h)

            # TODO: deterministic
            # g
            dist = self.pi_theta((h, r))
            g_int = dist.sample()
            outputs.g_int.append(g_int.float())
            outputs.g_probs.append(dist.probs)

            # g_loss
            i1, i2, i3 = self.decode(g_int)
            # assert (int(i1), int(i2), int(i3)) == \
            #        np.unravel_index(int(g_int), self.subtask_space)
            g2 = self.embed_task(i1, i2, i3).squeeze(1)
            g_loss = F.binary_cross_entropy(
                torch.clamp(g2, 0., 1.),
                r_target,
                reduction='none',
            )
            outputs.g_loss.append(torch.mean(g_loss, dim=-1, keepdim=True))
            g = interp(g, g2, c)
            outputs.g.append(g)

            # b
            dist = self.beta(torch.cat([conv_out, g], dim=-1))
            b = dist.sample().float()
            outputs.b_probs.append(dist.probs)

            # b_loss
            outputs.b_loss.append(-dist.log_probs(next_subtask[i]))
            outputs.b.append(b)

            # a
            g_broad = broadcast_3d(g, self.obs_shape[1:])
            conv_out2 = self.conv2((obs[i], g_broad))
            dist = self.actor(conv_out2)
            a = dist.sample()
            # a[:] = 'wsadeq'.index(input('act:'))

            outputs.a.append(a.float())
            outputs.a_probs.append(dist.probs)

            # v
            outputs.v.append(self.critic(conv_out2))

        stacked = []
        for x in outputs:
            stacked.append(torch.stack(x))

        hx = torch.cat(stacked, dim=-1)
        return hx, hx[-1]
