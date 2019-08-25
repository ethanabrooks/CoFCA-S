from collections import Counter

import torch
import torch.jit
from torch import nn as nn
from torch.nn import functional as F

import ppo.agent
from ppo.agent import AgentValues
from ppo.distributions import FixedCategorical
import numpy as np

# noinspection PyMissingConstructor
from ppo.events.recurrence import RecurrentState


class InteractivityAgent(nn.Module):
    def __init__(self, observation_space, **kwargs):
        nn.Module.__init__(self)
        obs_spaces = ppo.events.wrapper.Obs(**observation_space.spaces)
        assert obs_spaces.instructions.n == 2
        self.obs_shape = obs_spaces.base.shape
        self.obs_sections = [int(np.prod(s.shape)) for s in obs_spaces]
        self.agents = nn.ModuleList(
            [
                ppo.agent.Agent(obs_shape=obs_spaces.base.shape, **kwargs)
                for _ in range(3)
            ]
        )

    def parse_inputs(self, inputs: torch.Tensor):
        return ppo.events.wrapper.Obs(*torch.split(inputs, self.obs_sections, dim=-1))

    @property
    def recurrent_hidden_state_size(self):
        return self.agents[0].recurrent_hidden_state_size

    @property
    def is_recurrent(self):
        return self.agents[0].is_recurrent

    def forward(self, inputs, rnn_hxs, masks, deterministic=False, action=None):
        device = inputs.device
        N, _ = inputs.shape
        n = rnn_hxs.size(0)
        obs = ppo.events.wrapper.Obs(*torch.split(inputs, self.obs_sections, dim=-1))
        agent_idxs = (
            obs.instructions @ torch.arange(1, len(self.agents), device=device).float()
        ) - 1
        start_idxs = agent_idxs.view(inputs.size(0) // n, n)[0]

        base = obs.base.view(N, *self.obs_shape)
        combined_values = AgentValues(
            value=torch.zeros(N, 1),
            action=torch.zeros(N, 1, dtype=int),
            action_log_probs=torch.zeros(N, 1),
            rnn_hxs=torch.zeros(n, self.recurrent_hidden_state_size),
            dist=None,
            aux_loss=0,
            log=Counter(),
        )

        for i, agent in enumerate(self.agents):
            j = agent_idxs == i
            _j = start_idxs == i
            if not _j.any():
                continue
            agent_values = agent(
                base[j],
                rnn_hxs[_j],
                masks[j],
                deterministic=deterministic,
                action=None if action is None else action[j],
            )
            combined_values.value[j] = agent_values.value
            combined_values.action[j] = agent_values.action
            combined_values.action_log_probs[j] = agent_values.action_log_probs
            combined_values.rnn_hxs[_j] = agent_values.rnn_hxs
            combined_values.log.update(agent_values.log)

        for k in combined_values.log.keys():
            combined_values.log[k] /= N

        return combined_values

    def get_value(self, inputs, rnn_hxs, masks):
        return self(inputs, rnn_hxs, masks).value
