from collections import namedtuple

from gym.spaces import Box, Discrete
import torch
import torch.nn as nn

from ppo.agent import NNBase
from ppo.distributions import Categorical, DiagGaussian
from ppo.layers import Flatten
from ppo.utils import init, init_normc_, init_

AgentValues = namedtuple(
    "AgentValues", "value action action_log_probs aux_loss rnn_hxs log dist"
)


class Agent(nn.Module):
    def __init__(
        self,
        obs_shape,
        action_space,
        recurrent,
        hidden_size,
        entropy_coef,
        **network_args,
    ):
        super().__init__()
        self.entropy_coef = entropy_coef
        self.recurrent_module = MLPBase(
            obs_shape[0], recurrent=recurrent, hidden_size=hidden_size, **network_args
        )

        num_outputs = action_space.shape[0]
        self.dist = DiagGaussian(self.recurrent_module.output_size, num_outputs)
        self.continuous = isinstance(action_space, Box)

    @property
    def is_recurrent(self):
        return True

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.recurrent_module.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks, deterministic=False, action=None):
        value, actor_features, rnn_hxs = self.recurrent_module(inputs, rnn_hxs, masks)

        dist = self.dist(actor_features)

        if action is None:
            if deterministic:
                action = dist.mode()
            else:
                action = dist.sample()

        action_log_probs = dist.log_probs(action)
        entropy = dist.entropy().mean()
        return AgentValues(
            value=value,
            action=action,
            action_log_probs=action_log_probs,
            aux_loss=-self.entropy_coef * entropy,
            dist=dist,
            rnn_hxs=rnn_hxs,
            log=dict(entropy=entropy),
        )

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.recurrent_module(inputs, rnn_hxs, masks)
        return value


# noinspection PyMissingConstructor
class MLPBase(NNBase):
    def __init__(self, num_inputs, hidden_size, num_layers, recurrent, activation):
        nn.Module.__init__(self)
        recurrent_input_size = num_inputs
        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if self._recurrent:
            self.recurrent_module = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.recurrent_module.named_parameters():
                print("zeroed out", name)
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param)

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

        self.train()

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
