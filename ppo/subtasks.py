from collections import namedtuple

import numpy as np
import torch
import torch.jit
from gym.spaces import Box, Discrete
from torch import nn as nn
from torch.nn import functional as F

from ppo.agent import Agent, AgentValues, Flatten, NNBase
from ppo.distributions import Categorical, DiagGaussian
from ppo.utils import init
from ppo.wrappers import get_subtasks_obs_sections

RecurrentState = namedtuple(
    'RecurrentState', 'p r h g b log_prob '
                      'c_loss '
                      'l_loss '
                      'p_loss '
                      'r_loss '
                      'g_loss '
                      'b_loss '
                      'subtask')


class Concat(torch.jit.ScriptModule):
    def __init__(self, dim=-1):
        self.dim = dim
        super().__init__()

    def forward(self, input):
        return torch.cat(input, dim=self.dim)


class Reshape(torch.jit.ScriptModule):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


def init_(network, nonlinearity=None):
    if nonlinearity is None:
        return init(network,
                    nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
    return init(network,
                nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0),
                nn.init.calculate_gain(nonlinearity))


@torch.jit.script
def batch_conv1d(inputs, weights):
    outputs = []
    # one convolution per instance
    n = inputs.shape[0]
    for i in range(n):
        x = inputs[i]
        w = weights[i]
        outputs.append(
            F.conv1d(x.reshape(1, 1, -1), w.reshape(1, 1, -1), padding=1))
    return torch.cat(outputs)


@torch.jit.script
def interp(x1, x2, c):
    return c * x2.squeeze(1) + (1 - c) * x1


@torch.jit.script
def log_prob(i, probs):
    return torch.log(torch.gather(probs, -1, i))


class SubtasksTeacher(Agent):
    def __init__(self, task_space, obs_shape, **kwargs):
        self.obs_sections = get_subtasks_obs_sections(task_space)
        d, h, w = obs_shape
        assert d == sum(self.obs_sections)
        self.d = self.obs_sections.base + self.obs_sections.subtask
        super().__init__(obs_shape=(self.d, h, w), **kwargs)

    def preprocess_obs(self, inputs):
        batch_size, _, h, w = inputs.shape
        base_obs = inputs[:, :self.obs_sections.base]
        start = self.obs_sections.base
        stop = self.obs_sections.base + self.obs_sections.subtask
        subtask = inputs[:, start:stop, :, :]
        return torch.cat([base_obs, subtask], dim=1)

    def forward(self, inputs, *args, **kwargs):
        return super().forward(
            self.preprocess_obs(inputs), *args, **kwargs)

    def get_value(self, inputs, rnn_hxs, masks):
        return super().get_value(self.preprocess_obs(inputs), rnn_hxs, masks)


# noinspection PyMissingConstructor
class SubtasksAgent(Agent, NNBase):
    def __init__(self,
                 obs_shape,
                 action_space,
                 task_space,
                 hidden_size,
                 recurrent,
                 entropy_coef,
                 imitation_agent=None):
        nn.Module.__init__(self)
        self.imitation_agent = imitation_agent
        self.entropy_coef = entropy_coef
        self.obs_sections = get_subtasks_obs_sections(task_space)
        d, h, w = obs_shape
        assert d == sum(self.obs_sections)

        self.recurrent_module = SubtasksRecurrence(
            h=h,
            w=w,
            task_space=task_space,
            hidden_size=hidden_size,
            recurrent=recurrent,
        )

        self.conv1 = nn.Sequential(
            init_(
                nn.Conv2d(self.obs_sections.base, hidden_size, kernel_size=3, stride=1, padding=1),
                'relu'), nn.ReLU(), Flatten())

        conv_weight_shape = hidden_size, self.obs_sections.base, 3, 3
        self.conv_weight = nn.Sequential(
            nn.Linear(self.obs_sections.subtask,
                      np.prod(conv_weight_shape)),
            Reshape(-1, *conv_weight_shape))

        # self.conv2 = nn.Sequential(
        #     Concat(dim=1),
        #     init_(
        #         nn.Conv2d(self.obs_sections.base + self.obs_sections.subtask,
        #                   hidden_size, kernel_size=3, stride=1, padding=1),
        #         'relu'), nn.ReLU(), Flatten())

        input_size = (
                h * w * hidden_size +  # conv output
                sum(task_space.nvec[0]))  # task size

        if isinstance(action_space, Discrete):
            num_outputs = action_space.n
            actor = Categorical(input_size, num_outputs)
        elif isinstance(action_space, Box):
            num_outputs = action_space.shape[0]
            actor = DiagGaussian(input_size, num_outputs)
        else:
            raise NotImplementedError
        self.actor = nn.Sequential(Concat(), actor)

        self.critic = nn.Sequential(Concat(), init_(nn.Linear(input_size, 1)))

    @property
    def recurrent_hidden_state_size(self):
        return sum(self.recurrent_module.state_sizes)

    def get_hidden(self, inputs, rnn_hxs, masks):
        obs, _, task, next_subtask = torch.split(inputs, self.obs_sections, dim=1)
        task = task[:, :, 0, 0]
        next_subtask = next_subtask[:, :, 0, 0]

        # TODO: This is where we would embed the task if we were doing that

        conv_out = self.conv1(obs)
        recurrent_inputs = torch.cat([conv_out, task, next_subtask], dim=-1)
        x, rnn_hxs = self._forward_gru(recurrent_inputs, rnn_hxs, masks)
        hx = RecurrentState(*self.recurrent_module.parse_hidden(x))

        # multiplicative interaction
        weights = self.conv_weight(hx.g)
        outs = []
        for ob, weight in zip(obs, weights):
            outs.append(F.conv2d(ob.unsqueeze(0), weight, padding=(1, 1)))
        out = torch.cat(outs).view(*conv_out.shape)

        # _, _, h, w = obs.shape
        # g = hx.g.view(*hx.g.shape, 1, 1).expand(*hx.g.shape, h, w)
        # out = self.conv2((obs, g))

        return out, hx

    def forward(self, inputs, rnn_hxs, masks, action=None,
                deterministic=False):
        conv_out, hx = self.get_hidden(inputs, rnn_hxs, masks)
        dist = self.actor((conv_out, hx.g))

        if action is None:
            if deterministic:
                action = dist.mode()
            else:
                action = dist.sample()

        value = self.critic((conv_out, hx.g))
        action_log_probs = dist.log_probs(action)
        entropy = dist.entropy()
        # TODO: combine with other entropy?

        entropy_bonus = self.entropy_coef * entropy
        losses = dict(neg_action_log_prob=-action_log_probs)
        # losses = {k: v for k, v in hx._asdict().items() if k.endswith('_loss')}
        # losses.update(action_log_prob=action_log_probs)

        # self.recurrent_module.check_grad(**losses)
        aux_loss = -entropy_bonus
        if self.imitation_agent:
            imitation_dist = self.imitation_agent(inputs, rnn_hxs, masks).dist
            imitation_probs = imitation_dist.probs.detach().unsqueeze(1)
            log_probs = torch.log(dist.probs).unsqueeze(2)
            imitation_loss = imitation_probs @ log_probs
            aux_loss -= imitation_loss.view(-1)

        return AgentValues(
            value=value,
            action=action,
            action_log_probs=action_log_probs + hx.log_prob,
            aux_loss=aux_loss.mean(),
            rnn_hxs=rnn_hxs,
            dist=dist,
            log=losses)

    def get_value(self, inputs, rnn_hxs, masks):
        conv_out, hx = self.get_hidden(inputs, rnn_hxs, masks)
        return self.critic((conv_out, hx.g))

    @property
    def is_recurrent(self):
        return True


def trace(module_fn, in_size):
    return torch.jit.trace(
        module_fn(in_size), example_inputs=torch.rand(1, in_size))


class SubtasksRecurrence(torch.jit.ScriptModule):
    __constants__ = [
        'input_sections', 'subtask_space', 'state_sizes', 'recurrent'
    ]

    def __init__(self, h, w, task_space, hidden_size, recurrent):
        super().__init__()
        conv_out_size = h * w * hidden_size
        self.subtask_space = list(map(int, task_space.nvec[0]))
        subtask_size = sum(self.subtask_space)
        n_subtasks = task_space.shape[0]

        # networks
        self.recurrent = recurrent
        in_size = (
                conv_out_size +  # x
                subtask_size +  # r
                subtask_size +  # g
                1)  # b
        self.f = init_(nn.Linear(in_size, hidden_size))

        subcontroller = nn.GRUCell if recurrent else nn.Linear
        self.subcontroller = trace(
            lambda in_size: init_(subcontroller(in_size, hidden_size)),
            in_size=conv_out_size)  # h

        self.phi_update = trace(
            lambda in_size: init_(nn.Linear(in_size, 1), 'sigmoid'),
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

        self.pi_theta = trace(
            lambda in_size: nn.Sequential(
                init_(nn.Linear(in_size, np.prod(self.subtask_space))
                      ),  # all possible subtask specs
                nn.Softmax(dim=-1)),
            in_size=(
                    hidden_size +  # h
                    subtask_size))  # r

        self.beta = trace(
            lambda in_size: nn.Sequential(
                init_(nn.Linear(in_size, 2)),  # binary: done or not done
                nn.Softmax(dim=-1)),
            in_size=(
                    conv_out_size +  # x
                    subtask_size))  # g

        # embeddings
        for name, d in zip(
                ['type_embeddings', 'count_embeddings', 'obj_embeddings'],
                self.subtask_space):
            self.register_buffer(name, torch.eye(int(d)))

        self.register_buffer('l_values', torch.tensor([[1], [2]]))
        self.register_buffer('p_values', torch.eye(n_subtasks))

        task_sections = [n_subtasks] * task_space.nvec.shape[1]
        input_sections = [conv_out_size, *task_sections,
                          1]  # 1 for next_subtask
        self.input_sections = list(map(int, input_sections))
        state_sizes = RecurrentState(
            p=n_subtasks,
            r=subtask_size,
            h=hidden_size,
            g=subtask_size,
            b=1,
            log_prob=1,
            c_loss=1,
            l_loss=1,
            p_loss=1,
            r_loss=1,
            g_loss=1,
            b_loss=1,
            subtask=1)
        self.state_sizes = RecurrentState(*map(int, state_sizes))

    @torch.jit.script_method
    def parse_hidden(self, hx):
        return torch.split(hx, self.state_sizes, dim=-1)

    @torch.jit.script_method
    def embed_task(self, task_type, count, obj):
        return torch.cat([
            self.type_embeddings[task_type.long()],
            self.count_embeddings[count.long()],
            self.obj_embeddings[obj.long()],
        ],
            dim=-1)

    @torch.jit.script_method
    def forward(self, input, hx):
        assert hx is not None
        obs, task_type, count, obj, next_subtask = torch.split(
            input, self.input_sections, dim=-1)

        for x in task_type, count, obj, next_subtask:
            x.detach_()

        count -= 1
        M = self.embed_task(task_type[0], count[0], obj[0])
        p, r, h, g, b, _, _, _, _, _, _, _, float_subtask = self.parse_hidden(
            hx)

        if bool(torch.all(hx == 0)):  # new episode
            p[:, :, 0] = 1.  # initialize pointer to first subtask
            r[:] = M[:, 0]  # initialize r to first subtask
            g[:] = M[:, 0]  # initialize g to first subtask

        for x in p, r, h, g, b, float_subtask:
            x.squeeze_(0)

        ps = []
        rs = []
        hs = []
        gs = []
        bs = []
        log_probs = []
        c_losses = []
        l_losses = []
        p_losses = []
        r_losses = []
        g_losses = []
        b_losses = []
        subtasks = []

        n = obs.shape[0]
        # print('Recurrence: next_subtask', next_subtask)
        # if torch.any(next_subtask > 0):
        #     import ipdb; ipdb.set_trace()
        for i in range(n):
            float_subtask += next_subtask[i]
            subtasks.append(float_subtask)
            subtask = float_subtask.long()
            m = M.shape[0]

            s = self.f(torch.cat([obs[i], r, g, b], dim=-1))
            c = torch.sigmoid(self.phi_update(torch.cat([s, h], dim=-1)))

            # c_loss
            c_losses.append(
                F.binary_cross_entropy(c, next_subtask[i], reduction='none'))

            # TODO: figure this out
            # if self.recurrent:
            #     h2 = self.subcontroller(obs[i], h)
            # else:
            h2 = self.subcontroller(obs[i])

            l_logits = self.phi_shift(h2)
            l = F.softmax(l_logits, dim=1)

            # l_loss
            l_target = self.l_values[next_subtask[i].long()].view(-1)
            l_losses.append(
                F.cross_entropy(l_logits, l_target,
                                reduction='none').unsqueeze(1)  # TODO
            )

            p2 = batch_conv1d(p, l)

            # p_losss
            p_losses.append(
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
                r2.squeeze(1), r_target, reduction='none')
            r_losses.append(torch.mean(r_loss, dim=-1, keepdim=True))

            p = interp(p, p2, c)
            r = interp(r, r2, c)
            h = interp(h, h2, c)

            # TODO: deterministic
            # g
            probs = self.pi_theta(torch.cat([h, r], dim=-1))
            g_int = torch.multinomial(probs, 1)
            log_prob_g = log_prob(g_int, probs)

            # g_loss
            g_target = []
            for j in range(m):
                t1 = task_type[i, j, subtask[j]]
                t2 = count[i, j, subtask[j]]
                t3 = obj[i, j, subtask[j]]
                target_int = self.encode(t1, t2, t3)
                # assert target_int == np.ravel_multi_index((int(t1), int(t2), int(t3)),
                #                                           self.subtask_space)
                g_target.append(target_int)
            g_target = torch.stack(g_target)
            g_loss = log_prob(g_target, probs)
            g_losses.append(g_loss)

            # i1, i2, i3 = self.decode(g_int)
            i1, i2, i3 = self.decode(g_target)  # TODO
            # assert (int(i1), int(i2), int(i3)) == \
            #        np.unravel_index(int(g_int), self.subtask_space)
            g2 = self.embed_task(i1, i2, i3).squeeze(1)

            g = g2
            # g = interp(g, g2, c) # TODO

            # b
            probs = self.beta(torch.cat([obs[i], g], dim=-1))
            b = torch.multinomial(probs, 1)
            log_prob_b = log_prob(b, probs)
            b = b.float()

            # b_loss
            b_losses.append(log_prob(next_subtask[i].long(), probs))

            ps.append(p)
            rs.append(r)
            hs.append(h)
            gs.append(g)
            bs.append(b)
            log_probs.append(log_prob_g + log_prob_b)

        outs = []
        for x in (ps, rs, hs, gs, bs, log_probs, c_losses, l_losses, p_losses,
                  r_losses, g_losses, b_losses, subtasks):
            outs.append(torch.stack(x))

        # for k, out, state_size in zip(RecurrentState._fields, outs, self.state_sizes):
        #     out_shape = out.shape[2]
        #     if out_shape != state_size:
        #         print(k, out_shape, state_size)

        hx = torch.cat(outs, dim=-1)
        return hx, hx[-1]

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
