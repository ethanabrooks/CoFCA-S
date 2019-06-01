from collections import namedtuple

import torch
from gym import spaces
from gym.spaces import Discrete, Box
import numpy as np
from torch import nn as nn
from torch.nn import functional as F

from ppo.policy import NNBase, Flatten, Policy
from ppo.utils import init
from ppo.distributions import Categorical, DiagGaussian

RecurrentState = namedtuple('RecurrentState', 'p r h g a b v log_prob')


class Concat(nn.Module):
    def forward(self, *input):
        return torch.cat(input, dim=-1)


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


class SubtasksPolicy(NNBase):
    def __init__(self,
                 obs_shape,
                 action_shape,
                 task_shape,
                 hidden_size,
                 recurrent,
                 ):
        self.recurrent_module = SubtasksRecurrence(obs_shape=obs_shape,
                                                   action_shape=action_shape,
                                                   task_shape=task_shape,
                                                   hidden_size=hidden_size,
                                                   recurrent=recurrent
                                                   )

    def forward(self, inputs, rnn_hxs, masks, action=None, deterministic=False):
        x, rnn_hxs = self._forward_gru(inputs, rnn_hxs, masks)

        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        return value, action, action_log_probs, rnn_hxs

    def sample(self, logits):
        dist = torch.distributions.Categorical(logits=logits)
        x = dist.sample()
        return x, dist.log_prob(x)


class SubtasksRecurrence(nn.Module):
    def __init__(self, obs_shape, task_shape, hidden_size, recurrent,
                 conv, actor, critic):
        super().__init__()
        self.subtask_size, self.n_subtasks = task_shape
        self.obs_shape = h, w, d = obs_shape

        # TODO: look over all this
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.conv = (nn.Sequential(
            Reshape(*obs_shape),
            init_(nn.Conv2d(d, hidden_size, kernel_size=3, stride=1, padding=1)),
            nn.ReLU()))

        # todo: policy and critic

        self.recurrent = recurrent
        subcontroller = nn.GRUCell if recurrent else nn.Linear
        self.subcontroller = subcontroller(hidden_size * h * w,
                                           hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.phi_update = nn.Sequential(
            Concat(), init_(nn.Linear(2 * hidden_size, 1)))
        self.phi_shift = init_(nn.Linear(hidden_size, 3))  # 3 for {-1, 0, +1}

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('sigmoid'))

        conv_out_size = h * w * hidden_size
        self.f = nn.Sequential(
            Concat(),
            init_(nn.Linear(conv_out_size + 2 * self.subtask_size, hidden_size)))

    def split_hidden(self, hx):
        return RecurrentState(*torch.split(hx, RecurrentState(p=self.n_subtasks,
                                                              r=self.subtask_size,
                                                              h=self.hidden_size,
                                                              g=self.subtask_size,
                                                              a=self.action_size,
                                                              b=1,
                                                              v=1,
                                                              log_prob=1,
                                                              ), dim=-1))

    def forward(self, input, hx=None):
        assert hx is not None
        sections = [self.obs_size, self.n_subtasks * self.subtask_size]
        obs, task = torch.split(input, sections, dim=-1)
        M = task.view(-1, self.n_subtasks, self.task_size)
        hx = self.split_hidden(hx)

        # TODO: unzero zeroed values

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
            a, log_prob_a = self.sample(self.pi_phi(x, g))
            b, log_prob_b = self.sample(self.beta(x, g))
            v = self.value(x, g)
            log_prob = log_prob_g + log_prob_a + log_prob_b

            hx = RecurrentState(p=p, r=r, h=h, g=g, a=a, b=b, v=v, log_prob=log_prob)
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

#         self.base = SubtasksBase(*obs_shape, **network_args)
#
#     @staticmethod
#     def log_prob(dist, action, log_prob_g):
#         return dist.log_prob(action) + log_prob_g
#
#     def act(self, inputs, rnn_hxs, masks, deterministic=False):
#         value, actor_features, rnn_hxs, log_prob_g = self.base(inputs, rnn_hxs, masks)
#
#         dist = self.dist(actor_features)
#
#         if deterministic:
#             action = dist.mode()
#         else:
#             action = dist.sample()
#
#         return value, action, self.log_prob(dist, action, log_prob_g), rnn_hxs
#
#     def evaluate_actions(self, inputs, rnn_hxs, masks, action):
#         value, actor_features, rnn_hxs, log_prob_g = self.base(inputs, rnn_hxs, masks)
#         dist = self.dist(actor_features)
#
#         dist_entropy = dist.entropy().mean()
#
#         return value, self.log_prob(dist, action, log_prob_g), dist_entropy, rnn_hxs
#
#
# class SubtasksBase(NNBase):
#     def __init__(self, obs_shape, action_shape, task_shape, hidden_size):
#         self.obs_shape = h, w, d = obs_shape
#         self.task_shape = n_subtasks, subtask_size = task_shape
#         super().__init__(recurrent=True,
#                          recurrent_input_size=(obs_shape, task_shape),
#                          hidden_size=n_subtasks +  # p
#                                      subtask_size +  # r
#                                      hidden_size +  # h
#                                      subtask_size  # g
#                          )
#
#         init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
#                                constant_(x, 0), nn.init.calculate_gain('relu'))
#
#         # TODO: check this architecture
#         self.main = nn.Sequential(
#             Reshape(*obs_shape),
#             init_(
#                 nn.Conv2d(d, hidden_size, kernel_size=3, stride=1, padding=1)),
#             nn.ReLU(),
#             Flatten(),
#             init_(nn.Linear(hidden_size * h * w, hidden_size)),
#             nn.ReLU())
#
#         init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
#                                constant_(x, 0))
#
#         self.critic_linear = init_(nn.Linear(hidden_size, 1))
#         self.train()
#
#
#     def forward(self, inputs, rnn_hxs, masks):
#         obs, task = torch.split(inputs,
#                                 [np.prod(self.obs_shape), np.prod(self.task_shape)],
#                                 dim=-1)
#         conv_out = self.main(obs)
#
#         x = torch.cat([conv_out, task], dim=-1)
#         hx, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
#         p, r, h, g, log_prob_g = self.recurrent_module.split_hidden(hx)
#         x = torch.cat([conv_out, g], dim=-1)
#         return self.critic_linear(x), x, rnn_hxs, log_prob_g
#
#
# class MetaController(nn.Module):
#     def __init__(self, obs_shape, action_shape, task_shape, hidden_size):
