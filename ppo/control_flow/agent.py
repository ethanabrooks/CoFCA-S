from gym.spaces import MultiDiscrete
import torch
from torch import nn as nn
import torch.jit
from torch.nn import functional as F

from gridworld_env.subtasks_gridworld import Obs
import ppo
from ppo.agent import AgentValues, NNBase
import ppo.control_flow.lower_level
from ppo.control_flow.recurrence import Recurrence, RecurrentState
from ppo.control_flow.wrappers import Actions
from ppo.distributions import FixedCategorical


# noinspection PyMissingConstructor
class Agent(ppo.agent.Agent, NNBase):
    def __init__(
        self,
        obs_space,
        action_space,
        hidden_size,
        entropy_coef,
        hard_update,
        agent_load_path,
        agent_args,
        **kwargs,
    ):
        nn.Module.__init__(self)
        self.hard_update = hard_update
        self.entropy_coef = entropy_coef
        self.action_spaces = Actions(**action_space.spaces)
        self.obs_space = obs_space
        obs_spaces = Obs(**self.obs_space.spaces)
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
        return Recurrence(**kwargs)

    def forward(self, inputs, rnn_hxs, masks, action=None, deterministic=False):
        n = inputs.size(0)
        actions = None
        if action is not None:
            actions = Actions(
                *torch.split(action, [1] * len(self.action_spaces), dim=-1)
            )

        all_hxs, last_hx = self._forward_gru(
            inputs.view(n, -1), rnn_hxs, masks, actions=actions
        )
        rm = self.recurrent_module
        hx = RecurrentState(*rm.parse_hidden(all_hxs))

        if action is None:
            actions = Actions(a=hx.a, cg=hx.cg, cr=hx.cr, g=hx.g)

        if self.hard_update:
            dists = Actions(
                a=FixedCategorical(hx.a_probs),
                cg=FixedCategorical(hx.cg_probs),
                cr=FixedCategorical(hx.cr_probs),
                g=FixedCategorical(hx.g_probs),
            )
        else:
            dists = Actions(
                a=None
                if rm.agent  # use pre-trained agent so don't train
                else FixedCategorical(hx.a_probs),
                cg=None,
                cr=None,
                g=FixedCategorical(hx.g_probs),
            )

        log_probs = sum(
            dist.log_probs(a) for dist, a in zip(dists, actions) if dist is not None
        )
        entropies = sum(dist.entropy() for dist in dists if dist is not None)
        aux_loss = -self.entropy_coef * entropies.mean()
        log = {k: v for k, v in hx._asdict().items() if k.endswith("_loss")}

        return AgentValues(
            value=hx.v,
            action=torch.cat(actions, dim=-1),
            action_log_probs=log_probs,
            aux_loss=aux_loss,
            rnn_hxs=torch.cat(hx, dim=-1),
            dist=None,
            log=log,
        )

    def get_value(self, inputs, rnn_hxs, masks):
        n = inputs.size(0)
        all_hxs, last_hx = self._forward_gru(inputs.view(n, -1), rnn_hxs, masks)
        return self.recurrent_module.parse_hidden(all_hxs).v

    def _forward_gru(self, x, hxs, masks, actions=None):
        if actions is None:
            y = F.pad(x, [0, len(Actions._fields)], "constant", -1)
        else:
            y = torch.cat([x] + list(actions), dim=-1)
        return super()._forward_gru(y, hxs, masks)

    @property
    def recurrent_hidden_state_size(self):
        return sum(self.recurrent_module.state_sizes)

    @property
    def is_recurrent(self):
        return True
