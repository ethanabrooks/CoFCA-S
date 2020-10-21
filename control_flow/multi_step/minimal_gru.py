import torch
import torch.nn.functional as F
from torch import nn as nn

import control_flow.multi_step.abstract_recurrence as abstract_recurrence
from networks import AgentOutputs, MLP
from control_flow import Obs
from control_flow import optimal_padding
from control_flow import get_obs_sections
from distributions import Categorical
from utils import init_


class Agent(nn.Module, abstract_recurrence.Recurrence):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size,
        task_embed_size,
        conv_hidden_size,
        num_layers,
        entropy_coef,
        stride,
        kernel_size,
        lower_embed_size,
        **network_args,
    ):
        self.obs_spaces = Obs(**observation_space.spaces)
        nn.Module.__init__(self)
        abstract_recurrence.Recurrence.__init__(self)
        self.action_size = action_space.nvec.size
        self.entropy_coef = entropy_coef
        self.hidden_size = hidden_size
        self.task_embed_size = task_embed_size
        self.obs_sections = Obs(*get_obs_sections(self.obs_spaces))
        self.train_lines = len(self.obs_spaces.lines.nvec)

        # networks
        n_a = int(action_space.nvec[0])
        self.embed_task = self.build_embed_task(hidden_size)
        self.embed_action = nn.Embedding(n_a, hidden_size)
        self.critic = init_(nn.Linear(hidden_size, 1))
        d, h, w = observation_space.obs.shape
        padding = optimal_padding(kernel_size, stride)
        self.conv = nn.Conv2d(
            in_channels=d,
            out_channels=conv_hidden_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.embed_lower = nn.Embedding(
            self.action_space_nvec.lower + 1, lower_embed_size
        )

        self.dist = Categorical(hidden_size, n_a)
        network_args.update(recurrent=True, activation=nn.ReLU())
        self.recurrent_module = MLP(
            num_inputs=conv_hidden_size + self.train_lines * hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers + 1,
            **network_args,
        )

    @property
    def is_recurrent(self):
        return True

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.hidden_size

    def parse_inputs(self, inputs: torch.Tensor):
        return Obs(*torch.split(inputs, self.obs_sections, dim=-1))

    def forward(self, inputs, rnn_hxs, masks, deterministic=False, action=None):
        raw_inputs = inputs
        T, N, dim = raw_inputs.shape
        inputs = self.parse_input(raw_inputs)

        # parse non-action inputs
        state = Obs(*self.parse_obs(inputs.obs))
        state = state._replace(obs=state.obs.view(T, N, *self.obs_spaces.obs.shape))
        lines = state.lines.view(T, N, *self.obs_spaces.lines.shape)

        # build memory
        nl = len(self.obs_spaces.lines.nvec)
        M = self.embed_task(self.preprocess_embed(N, T, state)).view(
            N, -1, self.task_embed_size
        )
        value, actor_features, rnn_hxs = self.recurrent_module(x, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if action is None:
            if deterministic:
                action = dist.mode()
            else:
                action = dist.sample()
        else:
            action = action[:, 0]

        action_log_probs = dist.log_probs(action)
        entropy = dist.entropy().mean()
        action = F.pad(action, [0, self.action_size - 1])
        return AgentOutputs(
            value=value,
            action=action,
            action_log_probs=action_log_probs,
            aux_loss=-self.entropy_coef * entropy,
            dist=dist,
            rnn_hxs=rnn_hxs,
            log=dict(entropy=entropy),
        )

    # def preprocess(self, inputs):
    #     N, dim = inputs.shape
    #     inputs = self.parse_inputs(inputs)
    #     inputs = inputs._replace(obs=inputs.obs.view(N, *self.obs_spaces.obs.shape))
    #     # build memory
    #     nl, dl = self.obs_spaces.lines.shape
    #     lines = inputs.lines.long().view(N, nl, dl)
    #     lines = lines + self.offset
    #     M = self.embed_task(lines.view(N * nl, dl)).view(
    #         N, -1
    #     )  # n_batch, n_lines * hidden_size
    #     obs = (
    #         self.ll_conv(inputs.obs.permute(0, 2, 3, 1))
    #         .view(N, -1, self.conv_hidden_size)
    #         .max(dim=1)
    #         .values
    #     )
    #     return torch.cat([M, obs], dim=-1)

    def get_value(self, inputs, rnn_hxs, masks):
        x = self.preprocess(inputs)
        value, _, _ = self.recurrent_module(x, rnn_hxs, masks)
        return value
