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
    def forward(self, *input):
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


# noinspection PyMissingConstructor
class SubtasksAgent(Agent, NNBase):
    def __init__(self,
                 obs_shape,
                 action_space,
                 task_shape,
                 hidden_size,
                 recurrent,
                 ):
        nn.Module.__init__(self)
        self.recurrent_module = SubtasksRecurrence(*obs_shape,
                                                   *task_shape,
                                                   hidden_size=hidden_size,
                                                   recurrent=recurrent)

        _, self.task_size = task_shape
        d, h, w = obs_shape
        self.obs_size = np.prod(obs_shape)

        self.conv = (nn.Sequential(
            Reshape(*obs_shape),
            init_(nn.Conv2d(d, hidden_size, kernel_size=3, stride=1, padding=1), 'relu'),
            nn.ReLU()))

        input_size = (h * w * hidden_size +  # conv output
                      self.task_size)

        # TODO: multiplicative interaction stuff
        if isinstance(action_space, Discrete):
            num_outputs = action_space.n
            self.actor = Categorical(input_size, num_outputs)
        elif isinstance(action_space, Box):
            num_outputs = action_space.shape[0]
            self.actor = DiagGaussian(input_size, num_outputs)
        else:
            raise NotImplementedError

        self.critic = init_(nn.Linear(input_size, 1))

    @property
    def recurrent_hidden_state_size(self):
        return sum(self.recurrent_module.state_sizes)

    def parse_obs(self, inputs):
        # TODO: remove 'hints' in obs if they are there
        import ipdb; ipdb.set_trace()

    def forward(self, inputs, rnn_hxs, masks, action=None, deterministic=False):
        obs, task = self.parse_obs(inputs)
        # TODO: This is where we would embed the task if we were doing that
        conv_out = self.conv(obs)
        import ipdb;
        ipdb.set_trace()  # reshape and pack
        recurrent_inputs = torch.cat([conv_out, task])

        x, rnn_hxs = self._forward_gru(recurrent_inputs, rnn_hxs, masks)
        hx = self.recurrent_module.parse_hidden(rnn_hxs)
        dist = self.actor(conv_out, hx.g)

        if action is None:
            if deterministic:
                action = dist.mode()
            else:
                action = dist.sample()

        value = self.critic(conv_out, hx.g)
        action_log_probs = dist.log_probs(action) + hx.log_prob
        dist_entropy = dist.entropy().mean()  # TODO: combine with other entropy
        return value, action, action_log_probs, dist_entropy, rnn_hxs


class SubtasksRecurrence(nn.Module):
    def __init__(self, h, w, d, subtask_size, n_subtasks, hidden_size, recurrent):
        super().__init__()
        self.h = h
        self.w = w
        self.d = d
        self.n_subtasks = n_subtasks
        self.subtask_size = subtask_size

        # todo: policy and critic
        self.recurrent = recurrent
        subcontroller = nn.GRUCell if recurrent else nn.Linear
        self.subcontroller = subcontroller(hidden_size * h * w,
                                           hidden_size)

        self.phi_update = nn.Sequential(
            Concat(), init_(nn.Linear(2 * hidden_size, 1)))
        self.phi_shift = init_(nn.Linear(hidden_size, 3))  # 3 for {-1, 0, +1}

        conv_out_size = h * w * hidden_size
        self.f = nn.Sequential(
            Concat(),
            init_(nn.Linear(conv_out_size + 2 * self.subtask_size, hidden_size), 'sigmoid'))

        self.state_sizes = RecurrentState(p=self.n_subtasks, r=self.subtask_size, h=hidden_size,
                                          g=self.subtask_size, b=1,
                                          log_prob=1, )

    def sample(self, logits):
        dist = torch.distributions.Categorical(logits=logits)
        x = dist.sample()
        return x, dist.log_prob(x)

    def parse_hidden(self, hx):
        return RecurrentState(*torch.split(hx, self.state_sizes, dim=-1))

    def forward(self, input, hx=None):
        assert hx is not None
        sections = [self.obs_size, self.n_subtasks * self.subtask_size]
        obs, task = torch.split(input, sections, dim=-1)
        M = task.view(-1, self.n_subtasks, self.task_size)
        hx = self.parse_hidden(hx)

        # TODO: unzero zeroed values
        import ipdb;
        ipdb.set_trace()

        outputs = []
        for x in obs:
            x = self.conv(x)
            s = self.f(x, hx.r, hx.g, hx.b)
            c = F.sigmoid(self.phi_update(s, hx.h))
            h = self.subcontroller(s, hx.h)

            l = F.softmax(self.phi_shift(hx.h), dim=1)
            p = F.conv1d(hx.p, l)
            r = p @ M

            p, r, h = [c * x2 + (1 - c) * x1 for x1, x2 in
                       zip([hx.p, hx.r, hx.h], [p, r, h])]

            # TODO: deterministic
            g, log_prob_g = self.sample(self.pi_theta(h, r))
            g = c * g + (1 - c) * h.g
            b, log_prob_b = self.sample(self.beta(x, g))
            log_prob = log_prob_g + log_prob_b

            hx = RecurrentState(p=p, r=r, h=h, g=g, b=b, log_prob=log_prob)
            outputs.append(hx)

        hx = RecurrentState(*zip(*outputs))
        # TODO: pack and split these
        import ipdb;
        ipdb.set_trace()
        return outputs[-1], outputs


if __name__ == '__main__':
    x = SubtasksRecurrence(
        obs_shape=(10, 10, 18),
        task_shape=(5, 3),
        hidden_size=32,
        recurrent=True
    )
    import ipdb;

    ipdb.set_trace()
