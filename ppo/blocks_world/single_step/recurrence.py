from collections import namedtuple

from gym.spaces import Box
from torch import nn as nn

from ppo.agent import NNBase
from ppo.distributions import Categorical
from ppo.utils import init, init_normc_

RecurrentState = namedtuple("RecurrentState", "a probs v")
# "planned_probs plan v t state h model_loss"


class Recurrence(NNBase):
    def __init__(
        self, num_inputs, action_space, hidden_size, num_layers, recurrent, activation
    ):
        recurrent_module = nn.GRU if recurrent else None
        super(Recurrence, self).__init__(recurrent_module, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, init_normc_, lambda x: nn.init.constant_(x, 0))

        self.actor = nn.Sequential()
        self.critic = nn.Sequential()
        for i in range(num_layers):
            in_features = num_inputs if i == 0 else hidden_size
            self.actor.add_module(
                name=f"fc{i}",
                module=nn.Sequential(
                    init_(nn.Linear(in_features, hidden_size)), activation
                ),
            )
            self.critic.add_module(
                name=f"fc{i}",
                module=nn.Sequential(
                    init_(nn.Linear(in_features, hidden_size)), activation
                ),
            )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.dist = Categorical(self.output_size, action_space.n)
        self.continuous = isinstance(action_space, Box)

        self.train()

    @staticmethod
    def sample_new(x, dist):
        new = x < 0
        x[new] = dist.sample()[new].flatten()

    def forward(self, inputs, rnn_hxs, masks, action):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        dist = self.dist(hidden_actor)
        self.sample_new(action, dist)

        return RecurrentState(
            a=action, probs=dist.probs, v=self.critic_linear(hidden_critic)
        )
