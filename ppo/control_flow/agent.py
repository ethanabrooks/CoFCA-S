import torch
import torch.jit
from gym.spaces import MultiDiscrete
from torch import nn as nn
from torch.nn import functional as F

import ppo
import ppo.control_flow.lower_level
from gridworld_env.control_flow_gridworld import Obs
from ppo.agent import AgentValues, NNBase
from ppo.control_flow.lower_level import g_discrete_to_binary
from ppo.control_flow.recurrence import RecurrentState, DebugBase
from ppo.control_flow.wrappers import Actions
from ppo.distributions import FixedCategorical


class DebugAgent(ppo.agent.Agent, NNBase):
    def __init__(
        self,
        obs_space,
        action_space,
        hidden_size,
        g_entropy_coef,
        z_entropy_coef,
        l_entropy_coef,
        hard_update,
        agent_load_path,
        agent_args,
        **kwargs,
    ):
        nn.Module.__init__(self)
        self.hard_update = hard_update
        self.entropy_coefs = Actions(
            a=None,
            cg=None,
            cr=None,
            g=g_entropy_coef,
            z=z_entropy_coef,
            l=l_entropy_coef,
        )
        self.action_spaces = Actions(**action_space.spaces)
        self.obs_space = obs_space
        obs_spaces = Obs(**self.obs_space.spaces)
        self.n_subtasks = len(obs_spaces.subtasks.nvec)
        agent = None
        if agent_load_path is not None:
            agent = self.load_agent(
                agent_load_path=agent_load_path,
                action_spaces=self.action_spaces,
                obs_spaces=(
                    ppo.control_flow.lower_level.Obs(
                        base=obs_spaces.base,
                        subtask=MultiDiscrete(obs_spaces.subtasks.nvec[0, :3]),
                    )
                ),
                **agent_args,
            )
        self.recurrent_module = self.build_recurrent_module(
            agent=agent,
            hard_update=hard_update,
            hidden_size=hidden_size,
            obs_spaces=obs_spaces,
            action_spaces=self.action_spaces,
            activation=agent_args["activation"],
            **kwargs,
        )

    def load_agent(self, agent_load_path, device, **agent_args):
        agent = ppo.control_flow.LowerLevel(**agent_args)

        state_dict = torch.load(agent_load_path, map_location=device)
        assert "vec_normalize" not in state_dict, "oy"
        # state_dict["agent"].update(
        #     part0_one_hot=agent.part0_one_hot,
        #     part1_one_hot=agent.part1_one_hot,
        #     part2_one_hot=agent.part2_one_hot,
        # )
        agent.load_state_dict(state_dict["agent"])
        print(f"Loaded teacher parameters from {agent_load_path}.")
        return agent

    def build_recurrent_module(self, **kwargs):
        return DebugBase(**kwargs)
        # return Recurrence(**kwargs) TODO

    def forward(self, inputs, rnn_hxs, masks, deterministic=False, action=None):
        N = inputs.size(0)
        rm = self.recurrent_module
        all_hxs, last_hx = self._forward_gru(
            inputs.view(N, -1), rnn_hxs, masks, action=action
        )
        hx = RecurrentState(*rm.parse_hidden(last_hx))
        # hx = RecurrentState(*rm.parse_hidden(all_hxs)) # TODO what's the deal here?
        actions = Actions(a=hx.a, cg=hx.cg, cr=hx.cr, g=hx.g, z=hx.z, l=hx.l)
        dists = Actions(
            a=None
            if rm.agent  # use pre-trained agent so don't train
            else FixedCategorical(hx.a_probs),
            cg=None,
            cr=None,
            g=None,
            # g=FixedCategorical(hx.g_probs), TODO
            z=None,
            # z=FixedCategorical(
            #     hx.z_probs.view(N, self.n_subtasks, len(LineTypes._fields))
            # ), TODO
            l=FixedCategorical(hx.l_probs),
            # l=FixedCategorical(hx.l_probs) if self.hard_update else None, TODO
        )

        log_probs = sum(
            dist.log_probs(a).view(N, -1, 1).sum(1)
            for dist, a in zip(dists, actions)
            if dist is not None
        )
        entropies = Actions(
            *[None if dist is None else dist.entropy().mean() for dist in dists]
        )
        aux_loss = -sum(
            c * entropy
            for c, entropy in zip(self.entropy_coefs, entropies)
            if entropy is not None
        )
        log = {k: v for k, v in entropies._asdict().items() if v is not None}

        return AgentValues(
            value=hx.v,
            action=torch.cat(actions, dim=-1),
            action_log_probs=log_probs,
            aux_loss=aux_loss,
            dist=None,
            rnn_hxs=last_hx,
            log=log,
        )

    def get_value(self, inputs, rnn_hxs, masks):
        n = inputs.size(0)
        all_hxs, last_hx = self._forward_gru(inputs.view(n, -1), rnn_hxs, masks)
        return self.recurrent_module.parse_hidden(last_hx).v

    def _forward_gru(self, x, hxs, masks, action=None):
        if action is None:
            y = F.pad(x, [0, sum(self.recurrent_module.size_actions)], "constant", -1)
        else:
            y = torch.cat([x, action], dim=-1)
        return super()._forward_gru(y, hxs, masks)

    @property
    def recurrent_hidden_state_size(self):
        return sum(self.recurrent_module.state_sizes)

    @property
    def is_recurrent(self):
        return False  # TODO
