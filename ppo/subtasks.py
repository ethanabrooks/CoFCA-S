from collections import namedtuple

import torch
from gym import spaces
from gym.spaces import Discrete, Box
import numpy as np
from torch import nn as nn
from torch.nn import functional as F

from ppo.agent import NNBase, Flatten, Agent
from ppo.utils import init
from ppo.distributions import Categorical, DiagGaussian

RecurrentState = namedtuple('RecurrentState', 'p r h g b log_prob')


class Concat(nn.Module):
    def forward(self, input):
        return torch.cat(input, dim=-1)


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


def init_(network, nonlinearity=None):
    if nonlinearity is None:
        return init(network, nn.init.orthogonal_, lambda x: nn.init.
                    constant_(x, 0))
    return init(network, nn.init.orthogonal_, lambda x: nn.init.
                constant_(x, 0), nn.init.calculate_gain(nonlinearity))


def batch_conv1d(inputs, weights, padding):
    outputs = []
    # one convolution per instance
    for x, w in zip(inputs, weights):
        outputs.append(F.conv1d(x.reshape(1, 1, -1), w.reshape(1, 1, -1), padding=padding))
    return torch.cat(outputs)


def unravel_index3d(x, a, b, c):
    x1 = x // (b * c)
    r = x % (b * c)
    x2 = r // c
    x3 = r % c
    return x1, x2, x3


# noinspection PyMissingConstructor
class SubtasksAgent(Agent, NNBase):
    def __init__(self,
                 obs_shape,
                 action_space,
                 task_space,
                 hidden_size,
                 recurrent):
        nn.Module.__init__(self)
        n_subtasks, subtask_size = task_space.nvec.shape
        self.task_size = n_subtasks * subtask_size
        n_task_types = task_space.nvec[0, 0]
        self.n_cheat_layers = (n_task_types +  # task type one hot
                               1)  # + 1 for task objects
        d, h, w = obs_shape
        d -= self.task_size + self.n_cheat_layers
        obs_shape = d, h, w

        self.recurrent_module = SubtasksRecurrence(obs_shape=obs_shape,
                                                   task_space=task_space,
                                                   hidden_size=hidden_size,
                                                   recurrent=recurrent,
                                                   )

        self.obs_size = np.prod(obs_shape)

        self.conv = nn.Sequential(
            init_(nn.Conv2d(d, hidden_size, kernel_size=3, stride=1, padding=1), 'relu'),
            nn.ReLU(),
            Flatten()
        )

        input_size = (h * w * hidden_size +  # conv output
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
        self.actor = nn.Sequential(
            Concat(), actor
        )

        self.critic = nn.Sequential(
            Concat(),
            init_(nn.Linear(input_size, 1)))

    @property
    def recurrent_hidden_state_size(self):
        return sum(self.recurrent_module.state_sizes)

    def parse_obs(self, inputs):
        obs = inputs[:, :-(self.task_size + self.n_cheat_layers)]
        task = inputs[:, -self.task_size:, 0, 0]
        return obs, task

    def get_hidden(self, inputs, rnn_hxs, masks):
        obs, task = self.parse_obs(inputs)
        # TODO: This is where we would embed the task if we were doing that
        conv_out = self.conv(obs)

        recurrent_inputs = torch.cat([conv_out, task], dim=-1)
        x, rnn_hxs = self._forward_gru(recurrent_inputs, rnn_hxs, masks)
        return conv_out, self.recurrent_module.parse_hidden(x)

    def forward(self, inputs, rnn_hxs, masks, action=None, deterministic=False):
        conv_out, hx = self.get_hidden(inputs, rnn_hxs, masks)
        dist = self.actor((conv_out, hx.g))

        if action is None:
            if deterministic:
                action = dist.mode()
            else:
                action = dist.sample()

        value = self.critic((conv_out, hx.g))
        action_log_probs = dist.log_probs(action) + hx.log_prob
        dist_entropy = dist.entropy().mean()  # TODO: combine with other entropy
        return value, action, action_log_probs, dist_entropy, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        conv_out, hx = self.get_hidden(inputs, rnn_hxs, masks)
        return self.critic((conv_out, hx.g))

    @property
    def is_recurrent(self):
        return True


class SubtasksRecurrence(nn.Module):
    def __init__(self, obs_shape, task_space, hidden_size, recurrent):
        super().__init__()
        self.d, self.h, self.w = d, h, w = obs_shape
        self.subtask_space = subtask_space = task_space.nvec[0]
        self.n_subtasks, _ = task_space.nvec.shape
        self.subtask_size = np.sum(self.subtask_space)
        self.conv_out_size = h * w * hidden_size

        # networks
        self.recurrent = recurrent
        self.f = nn.Sequential(
            Concat(),
            init_(nn.Linear(self.conv_out_size +  # x
                            self.subtask_size +  # r
                            self.subtask_size +  # g
                            1  # b
                            , hidden_size), 'sigmoid'))

        subcontroller = nn.GRUCell if recurrent else nn.Linear
        self.subcontroller = nn.Sequential(
            Concat(),
            init_(subcontroller(hidden_size +  # s
                                hidden_size,  # h
                                hidden_size))
        )

        self.phi_update = nn.Sequential(
            Concat(), init_(nn.Linear(2 * hidden_size, 1)))
        self.phi_shift = init_(nn.Linear(hidden_size, 3))  # 3 for {-1, 0, +1}
        self.pi_theta = nn.Sequential(
            Concat(),
            Categorical(hidden_size +  # h
                        self.subtask_size,  # r
                        np.prod(subtask_space)  # all possible subtask specs
                        )
        )
        self.beta = nn.Sequential(
            Concat(),
            Categorical(self.conv_out_size +  # x
                        self.subtask_size,  # g
                        2  # binary: done or not done
                        )
        )

        # embeddings
        self.embeddings = nn.ParameterList(
            [nn.Parameter(torch.eye(d.item()), requires_grad=False)
             for d in subtask_space]
        )

        self.state_sizes = RecurrentState(p=self.n_subtasks, r=self.subtask_size, h=hidden_size,
                                          g=self.subtask_size, b=1,
                                          log_prob=1, )

    def parse_hidden(self, hx):
        return RecurrentState(*torch.split(hx, self.state_sizes, dim=-1))

    def embed_task(self, *task_codes):
        one_hots = [e[t.long()]
                    for e, t in
                    zip(self.embeddings, task_codes)]
        return torch.cat(one_hots, dim=-1)

    def forward(self, input, hx=None):
        assert hx is not None
        sections = [self.conv_out_size] + [self.n_subtasks] * 3
        obs, task_type, count, obj = torch.split(input, sections, dim=-1)

        count -= 1
        M = self.embed_task(task_type[0], count[0], obj[0])
        # TODO: why are both tasks the same?

        new_episode = torch.all(hx == 0)
        hx = self.parse_hidden(hx)
        if new_episode:
            hx.p[:, :, 0] = 1  # initialize pointer to first subtask
            hx.r[:] = hx.g[:] = M[:, 0]  # initialize r/g to first subtask

        # TODO: integrate this into parse_hidden
        for x in hx:
            x.squeeze_(0)

        outputs = []
        for x in obs:
            s = self.f((x, hx.r, hx.g, hx.b))
            c = torch.sigmoid(self.phi_update((s, hx.h)))
            h = self.subcontroller((s, hx.h))

            l = F.softmax(self.phi_shift(hx.h), dim=1)
            p = batch_conv1d(hx.p, l, padding=1)
            r = p @ M

            p, r, h = [c * x2.squeeze(1) + (1 - c) * x1
                       for x1, x2 in
                       zip([hx.p, hx.r, hx.h], [p, r, h])]

            # TODO: deterministic
            # g
            dist = self.pi_theta((h, r))
            g = dist.sample()
            log_prob_g = dist.log_probs(g)

            idxs = unravel_index3d(g, *self.subtask_space)
            g = self.embed_task(*idxs).squeeze(1)
            g = c * g + (1 - c) * hx.g

            # b
            dist = self.beta((x, g))
            b = dist.sample().float()
            log_prob_b = dist.log_probs(b)

            hx = RecurrentState(p=p, r=r, h=h, g=g, b=b,
                                log_prob=(log_prob_g + log_prob_b))
            outputs.append(hx)

        stacked = list(map(torch.stack, zip(*outputs)))
        hx = torch.cat(stacked, dim=-1)
        return hx, hx[-1]
