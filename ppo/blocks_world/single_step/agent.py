from ppo.agent import AgentValues

# noinspection PyMissingConstructor
from ppo.blocks_world.single_step.recurrence import MLPBase

from collections import namedtuple

from gym.spaces import Box, Discrete
import torch.nn as nn

from ppo.distributions import Categorical, DiagGaussian

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
        super(Agent, self).__init__()
        self.entropy_coef = entropy_coef
        if len(obs_shape) == 1:
            self.recurrent_module = MLPBase(
                obs_shape[0],
                recurrent=recurrent,
                hidden_size=hidden_size,
                **network_args,
            )
        else:
            raise NotImplementedError

        if isinstance(action_space, Discrete):
            num_outputs = action_space.n
            self.dist = Categorical(self.recurrent_module.output_size, num_outputs)
        elif isinstance(action_space, Box):
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.recurrent_module.output_size, num_outputs)
        else:
            raise NotImplementedError
        self.continuous = isinstance(action_space, Box)

    @property
    def is_recurrent(self):
        return self.recurrent_module.is_recurrent

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


# class Agent(ppo.agent.Agent, NNBase):
#     def __init__(self, entropy_coef, model_loss_coef, recurrence):
#         nn.Module.__init__(self)
#         self.model_loss_coef = model_loss_coef
#         self.entropy_coef = entropy_coef
#         self.recurrent_module = recurrence
#
#     @property
#     def recurrent_hidden_state_size(self):
#         return sum(self.recurrent_module.state_sizes)
#
#     @property
#     def is_recurrent(self):
#         return True
#
#     def forward(self, inputs, rnn_hxs, masks, deterministic=False, action=None):
#         N = inputs.size(0)
#         all_hxs, last_hx = self._forward_gru(
#             inputs.view(N, -1), rnn_hxs, masks, action=action
#         )
#         rm = self.recurrent_module
#         hx = rm.parse_hidden(all_hxs)
#         dist = FixedCategorical(hx.probs)
#         action_log_probs = dist.log_probs(hx.a)
#         entropy = dist.entropy()
#         aux_loss = self.model_loss_coef * hx.model_loss - self.entropy_coef * entropy
#         return AgentValues(
#             value=hx.v,
#             action=hx.a,
#             action_log_probs=action_log_probs,
#             aux_loss=aux_loss.mean(),
#             dist=dist,
#             rnn_hxs=last_hx,
#             log=dict(entropy=entropy, model_loss=hx.model_loss.mean()),
#         )
#
#     def _forward_gru(self, x, hxs, masks, action=None):
#         if action is None:
#             y = F.pad(x, [0, self.recurrent_module.action_size], "constant", -1)
#         else:
#             y = torch.cat([x, action.float()], dim=-1)
#         return super()._forward_gru(y, hxs, masks)
#
#     def get_value(self, inputs, rnn_hxs, masks):
#         all_hxs, last_hx = self._forward_gru(
#             inputs.view(inputs.size(0), -1), rnn_hxs, masks
#         )
#         return self.recurrent_module.parse_hidden(last_hx).v
