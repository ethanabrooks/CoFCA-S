from collections import namedtuple, Hashable
from dataclasses import replace, dataclass, astuple

import torch
import torch.nn.functional as F

import our_recurrence
from data_types import Obs, RawAction
from distributions import FixedCategorical

AgentOutputs = namedtuple(
    "AgentValues", "value action action_log_probs aux_loss rnn_hxs log dist"
)


@dataclass
class Agent(our_recurrence.Recurrence):
    entropy_coef: float

    def __hash__(self):
        return hash(tuple(x for x in astuple(self) if isinstance(x, Hashable)))

    @property
    def is_recurrent(self):
        return False

    # noinspection PyMethodOverriding
    def forward(
        self, inputs, rnn_hxs, masks, deterministic=False, action=None, **kwargs
    ):

        N, dim = inputs.shape

        # parse non-action inputs
        state = Obs(*torch.split(inputs, self.obs_sections, dim=-1))
        state = replace(state, obs=state.obs.view(N, *self.obs_spaces.obs.shape))
        lines = state.lines.view(N, *self.obs_spaces.lines.shape).long()

        # build memory
        nl = len(self.obs_spaces.lines.nvec)
        assert nl == 1
        M = self.embed_task(lines.view(-1, self.obs_spaces.lines.nvec[0].size)).view(
            N, self.task_embed_size
        )

        h1 = self.conv(state.obs)
        resources = self.embed_resources(state.resources)
        next_actions = self.embed_next_action(state.next_actions.long()).view(N, -1)
        embedded_lower = self.embed_lower(
            state.partial_action.long()
        )  # +1 to deal with negatives
        zeta1_input = torch.cat(
            [M, h1, resources, embedded_lower, next_actions], dim=-1
        )
        z1 = F.relu(self.zeta1(zeta1_input))

        value = self.critic(z1)

        a_logits = self.actor(z1) - state.action_mask * 1e10
        dist = FixedCategorical(logits=a_logits)

        if action is None:
            action = dist.sample()
        else:
            action = action[:, 0]

        action_log_probs = dist.log_probs(action)
        entropy = dist.entropy().mean()
        action = RawAction(
            delta=torch.zeros_like(action),
            dg=torch.zeros_like(action),
            ptr=torch.zeros_like(action),
            a=action,
        )
        return AgentOutputs(
            value=value,
            action=torch.cat(astuple(action), dim=-1),
            action_log_probs=action_log_probs,
            aux_loss=-self.entropy_coef * entropy,
            dist=dist,
            rnn_hxs=rnn_hxs,
            log=dict(entropy=entropy),
        )

    @property
    def recurrent_hidden_state_size(self):
        return 1

    def get_value(self, inputs, rnn_hxs, masks):
        return self.forward(inputs, rnn_hxs, masks).value
