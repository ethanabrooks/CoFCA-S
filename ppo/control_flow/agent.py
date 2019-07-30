from gym.spaces import MultiDiscrete, Discrete, Box
import torch
from torch import nn as nn
import torch.jit
from torch.nn import functional as F

from gridworld_env.control_flow_gridworld import LineTypes, Obs
import ppo
from ppo.agent import AgentValues, NNBase, CNNBase
import ppo.control_flow.lower_level
from ppo.control_flow.recurrence import Recurrence, RecurrentState
from ppo.control_flow.wrappers import Actions
from ppo.distributions import FixedCategorical, Categorical, DiagGaussian
from ppo.utils import init, init_normc_


class DebugAgent(nn.Module):
    def __init__(
        self, device, agent_args, obs_space, action_space, l_entropy_coef, **kwargs
    ):
        super().__init__()
        # super().__init__(
        #     obs_shape=obs_space.shape, action_space=action_space, **agent_args
        # )
        obs_shape = obs_space.shape
        entropy_coef = agent_args["entropy_coef"]
        recurrent = agent_args["recurrent"]
        hidden_size = agent_args["hidden_size"]
        self.entropy_coef = entropy_coef
        if len(obs_shape) == 3:
            self.base = CNNBase(
                *obs_shape, recurrent=recurrent, hidden_size=hidden_size
            )
        elif len(obs_shape) == 1:
            self.base = MLPBase(
                obs_shape[0],
                recurrent=recurrent,
                hidden_size=agent_args["hidden_size"],
                num_layers=agent_args["num_layers"],
                activation=agent_args["activation"],
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


class MLPBase(NNBase):
    def __init__(self, num_inputs, hidden_size, num_layers, recurrent, activation):
        recurrent_module = nn.GRU if recurrent else None
        super(MLPBase, self).__init__(recurrent_module, num_inputs, hidden_size)

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

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


# noinspection PyMissingConstructor
class Agent(ppo.agent.Agent, NNBase):
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
        return Recurrence(**kwargs)

    def forward(self, inputs, rnn_hxs, masks, action=None, deterministic=False):
        N = inputs.size(0)
        actions = None
        rm = self.recurrent_module
        if action is not None:
            actions = Actions(*torch.split(action, rm.size_actions, dim=-1))

        all_hxs, last_hx = self._forward_gru(
            inputs.view(N, -1), rnn_hxs, masks, actions=actions
        )
        hx = RecurrentState(*rm.parse_hidden(all_hxs))

        if action is None:
            actions = Actions(a=hx.a, cg=hx.cg, cr=hx.cr, g=hx.g, z=hx.z, l=hx.l)

        dists = Actions(
            a=None
            if rm.agent  # use pre-trained agent so don't train
            else FixedCategorical(hx.a_probs),
            cg=None,
            cr=None,
            g=FixedCategorical(hx.g_probs),
            z=FixedCategorical(
                hx.z_probs.view(N, self.n_subtasks, len(LineTypes._fields))
            ),
            l=FixedCategorical(hx.l_probs) if self.hard_update else None,
        )

        log_probs = sum(
            dist.log_probs(a).view(N, -1, 1).sum(1)
            for dist, a in zip(dists, actions)
            if dist is not None
        )
        # log_probs = log_probs + z_dist.log_probs(actions.z).sum(1)
        entropies = sum(
            c * dist.entropy().view(N, -1)
            for c, dist in zip(self.entropy_coefs, dists)
            if dist is not None
        )
        aux_loss = -entropies.mean()

        return AgentValues(
            value=hx.v,
            action=torch.cat(actions, dim=-1),
            action_log_probs=log_probs,
            aux_loss=aux_loss,
            rnn_hxs=torch.cat(hx, dim=-1),
            dist=None,
            log={},
        )

    def get_value(self, inputs, rnn_hxs, masks):
        n = inputs.size(0)
        all_hxs, last_hx = self._forward_gru(inputs.view(n, -1), rnn_hxs, masks)
        return self.recurrent_module.parse_hidden(all_hxs).v

    def _forward_gru(self, x, hxs, masks, actions=None):
        if actions is None:
            y = F.pad(x, [0, sum(self.recurrent_module.size_actions)], "constant", -1)
        else:
            y = torch.cat([x] + list(actions), dim=-1)
        return super()._forward_gru(y, hxs, masks)

    @property
    def recurrent_hidden_state_size(self):
        return sum(self.recurrent_module.state_sizes)

    @property
    def is_recurrent(self):
        return True
