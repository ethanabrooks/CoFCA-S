from gym.spaces import Box, Discrete

from gym.spaces import Box, Discrete
from torch import nn as nn

from ppo.agent import AgentValues
from ppo.agent import CNNBase, MLPBase
from ppo.distributions import Categorical
from ppo.distributions import DiagGaussian


# noinspection PyMissingConstructor


class DebugAgent(nn.Module):
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
        if len(obs_shape) == 3:
            self.base = CNNBase(
                *obs_shape, recurrent=recurrent, hidden_size=hidden_size, **network_args
            )
        elif len(obs_shape) == 1:
            self.base = MLPBase(
                obs_shape[0],
                recurrent=recurrent,
                hidden_size=hidden_size,
                **network_args,
            )
        else:
            raise NotImplementedError

        if isinstance(action_space, Discrete):
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif isinstance(action_space, Box):
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError
        self.continuous = isinstance(action_space, Box)

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks, deterministic=False, action=None):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)

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
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value
