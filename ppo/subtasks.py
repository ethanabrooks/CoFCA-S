from collections import namedtuple

from gym.spaces import Box, Discrete
import numpy as np
import torch
from torch import nn as nn
import torch.jit
from torch.nn import functional as F

from ppo.agent import Agent, Flatten, NNBase
from ppo.distributions import Categorical, DiagGaussian
from ppo.utils import init

RecurrentState = namedtuple('RecurrentState', 'p r h g b log_prob aux_loss')


class Concat(torch.jit.ScriptModule):
    def forward(self, input):
        return torch.cat(input, dim=-1)


class Reshape(torch.jit.ScriptModule):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


def init_(network, nonlinearity=None):
    if nonlinearity is None:
        return init(network, nn.init.orthogonal_,
                    lambda x: nn.init.constant_(x, 0))
    return init(network, nn.init.orthogonal_,
                lambda x: nn.init.constant_(x, 0),
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


# noinspection PyMissingConstructor
class SubtasksAgent(Agent, NNBase):
    def __init__(self, obs_shape, action_space, task_space, hidden_size,
                 recurrent, entropy_coef):
        nn.Module.__init__(self)
        self.entropy_coef = entropy_coef
        n_subtasks, subtask_size = task_space.nvec.shape
        self.task_size = n_subtasks * subtask_size
        n_task_types = task_space.nvec[0, 0]
        self.n_cheat_layers = (
                n_task_types +  # task type one hot
                1 +  # task objects
                1)  # iterate
        d, h, w = obs_shape
        d -= self.task_size + self.n_cheat_layers
        obs_shape = d, h, w

        self.recurrent_module = SubtasksRecurrence(
            obs_shape=obs_shape,
            task_space=task_space,
            hidden_size=hidden_size,
            recurrent=recurrent,
        )

        self.obs_size = np.prod(obs_shape)

        self.conv = nn.Sequential(
            init_(
                nn.Conv2d(d, hidden_size, kernel_size=3, stride=1, padding=1),
                'relu'), nn.ReLU(), Flatten())

        input_size = (
                h * w * hidden_size +  # conv output
                sum(task_space.nvec[0]))  # task size

        # TODO: multiplicative interaction stuff
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
        obs = inputs[:, :-(self.task_size + self.n_cheat_layers)]
        task = inputs[:, -self.task_size:, 0, 0]
        iterate = inputs[:, -1:, 0, 0]

        # TODO: This is where we would embed the task if we were doing that

        conv_out = self.conv(obs)
        recurrent_inputs = torch.cat([conv_out, task, iterate], dim=-1)
        x, rnn_hxs = self._forward_gru(recurrent_inputs, rnn_hxs, masks)
        return conv_out, RecurrentState(*self.recurrent_module.parse_hidden(x))

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
        action_log_probs = dist.log_probs(action) + hx.log_prob
        dist_entropy = dist.entropy().mean(
        )  # TODO: combine with other entropy
        return value, action, action_log_probs, dist_entropy, rnn_hxs

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, action_log_probs, entropy_bonus, rnn_hxs, log_values = \
            super().evaluate_actions(inputs, rnn_hxs, masks, action)
        aux_loss = -action_log_probs.mean() - entropy_bonus
        return value, action_log_probs, aux_loss, rnn_hxs, log_values

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
    __constants__ = ['input_sections', 'subtask_space', 'state_sizes', 'recurrent']

    def __init__(self, obs_shape, task_space, hidden_size, recurrent):
        super().__init__()
        self.d, self.h, self.w = d, h, w = obs_shape
        self.subtask_space = subtask_space = list(map(int, task_space.nvec[0]))
        self.n_subtasks, _ = task_space.nvec.shape
        self.subtask_size = int(np.sum(self.subtask_space))
        conv_out_size = h * w * hidden_size
        input_sections = [conv_out_size] + [self.n_subtasks] * 3 + [1]
        self.input_sections = [int(n) for n in input_sections]

        # networks
        self.recurrent = recurrent
        in_size = (
                conv_out_size +  # x
                self.subtask_size +  # r
                self.subtask_size +  # g
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
                init_(nn.Linear(in_size, np.prod(subtask_space))),  # all possible subtask specs
                nn.Softmax(dim=-1)),
            in_size=(
                    hidden_size +  # h
                    self.subtask_size))  # r

        self.beta = trace(
            lambda in_size: nn.Sequential(
                init_(nn.Linear(in_size, 2)),  # binary: done or not done
                nn.Softmax(dim=-1)),
            in_size=(
                    conv_out_size +  # x
                    self.subtask_size))  # g

        # embeddings
        self.type_embeddings, self.count_embeddings, self.obj_embeddings = [
            nn.Parameter(torch.eye(d), requires_grad=False)
            for d in subtask_space
        ]

        self.state_sizes = RecurrentState(
            p=self.n_subtasks,
            r=self.subtask_size,
            h=hidden_size,
            g=self.subtask_size,
            b=1,
            log_prob=1,
            aux_loss=1,
        )

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
        obs, task_type, count, obj, iterate = torch.split(
            input, self.input_sections, dim=-1)

        for x in task_type, count, obj, iterate:
            x.detach_()

        count -= 1
        M = self.embed_task(task_type[0], count[0], obj[0])
        # TODO: why are both tasks the same?

        p, r, h, g, b, _, _ = self.parse_hidden(hx)
        if bool(torch.all(hx == 0)):  # new episode
            p[:, :, 0] = 1.  # initialize pointer to first subtask
            r[:] = M[:, 0]  # initialize r to first subtask
            g[:] = M[:, 0]  # initialize g to first subtask

        for x in p, r, h, g, b:
            x.squeeze_(0)

        ps = []
        rs = []
        hs = []
        gs = []
        bs = []
        log_probs = []
        aux_losses = []

        n = obs.shape[0]
        for i in range(n):
            s = self.f(torch.cat([obs[i], r, g, b], dim=-1))

            c = torch.sigmoid(self.phi_update(torch.cat([s, h], dim=-1)))

            aux_loss = F.binary_cross_entropy(c, iterate[i], reduction='none') # TODO

            # if self.recurrent:
            #     h2 = self.subcontroller(obs[i], h)
            # else:
            h2 = self.subcontroller(obs[i])
            # TODO: this would not work for GRU (recurrent)

            l = F.softmax(self.phi_shift(h2), dim=1)
            p2 = batch_conv1d(p, l)
            r2 = p2 @ M

            p = interp(p, p2, c)
            r = interp(r, r2, c)
            h = interp(h, h2, c)

            # TODO: deterministic
            # g
            probs = self.pi_theta(torch.cat([h, r], dim=-1))
            g_int = torch.multinomial(probs, 1)
            log_prob_g = torch.log(torch.gather(probs, -1, g_int))

            i1, i2, i3 = self.unrave_index_subtask_space(g_int)
            g2 = self.embed_task(i1, i2, i3).squeeze(1)
            g = interp(g, g2, c)

            # b
            probs = self.beta(torch.cat([obs[i], g], dim=-1))
            b = torch.multinomial(probs, 1)
            log_prob_b = torch.log(torch.gather(probs, -1, b))
            b = b.float()

            ps.append(p)
            rs.append(r)
            hs.append(h)
            gs.append(g)
            bs.append(b)
            log_probs.append(log_prob_g + log_prob_b)
            aux_losses.append(aux_loss)

        outs = []
        for x in ps, rs, hs, gs, bs, log_probs, aux_losses:
            outs.append(torch.stack(x))

        hx = torch.cat(outs, dim=-1)
        return hx, hx[-1]

    def unrave_index_subtask_space(self, g):
        x1, x2, x3 = self.subtask_space
        g1 = g / (x2 * x3)
        x4 = g % (x2 * x3)
        g2 = x4 / x3
        g3 = x4 % x3
        return g1, g2, g3
