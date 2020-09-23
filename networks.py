from collections import namedtuple

from gym.spaces import Box, Discrete
import torch
import torch.nn as nn

from distributions import Categorical, DiagGaussian
from layers import Flatten
from utils import init, init_normc_, init_

AgentOutputs = namedtuple(
    "AgentValues", "value action action_log_probs aux_loss rnn_hxs " "log dist"
)


class Agent(nn.Module):
    def __init__(
        self,
        obs_spaces,
        action_space,
        recurrent,
        hidden_size,
        entropy_coef,
        **network_args,
    ):
        super(Agent, self).__init__()
        self.entropy_coef = entropy_coef
        self.recurrent_module = self.build_recurrent_module(
            hidden_size, network_args, obs_spaces, recurrent
        )

        if isinstance(action_space, Discrete):
            num_outputs = action_space.n
            self.dist = Categorical(self.recurrent_module.output_size, num_outputs)
        elif isinstance(action_space, Box):
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.recurrent_module.output_size, num_outputs)
        else:
            raise NotImplementedError
        self.continuous = isinstance(action_space, Box)

    def build_recurrent_module(self, hidden_size, network_args, obs_spaces, recurrent):
        if len(obs_spaces) == 3:
            return CNNBase(
                *obs_spaces,
                recurrent=recurrent,
                hidden_size=hidden_size,
                **network_args,
            )
        elif len(obs_spaces) == 1:
            return MLPBase(
                obs_spaces[0],
                recurrent=recurrent,
                hidden_size=hidden_size,
                **network_args,
            )
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.recurrent_module.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.recurrent_module.recurrent_hidden_state_size

    def forward(
        self, inputs, rnn_hxs, masks, deterministic=False, action=None, **kwargs
    ):
        value, actor_features, rnn_hxs = self.recurrent_module(
            inputs, rnn_hxs, masks, **kwargs
        )

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
        return AgentOutputs(
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


class NNBase(nn.Module):
    def __init__(self, recurrent: bool, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if self._recurrent:
            self.recurrent_module = self.build_recurrent_module(
                recurrent_input_size, hidden_size
            )
            for name, param in self.recurrent_module.named_parameters():
                print("zeroed out", name)
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param)

    def build_recurrent_module(self, input_size, hidden_size):
        return nn.GRU(input_size, hidden_size)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.recurrent_module(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, *x.shape[1:])

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.recurrent_module(
                    x[start_idx:end_idx], hxs * masks[start_idx].view(1, -1, 1)
                )

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, d, h, w, activation, hidden_size, num_layers, recurrent=False):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        self.main = nn.Sequential(
            init_(nn.Conv2d(d, hidden_size, kernel_size=1)),
            activation,
            *[
                nn.Sequential(
                    init_(
                        nn.Conv2d(hidden_size, hidden_size, kernel_size=1), activation
                    ),
                    activation,
                )
                for _ in range(num_layers)
            ],
            # init_(nn.Conv2d(d, 32, 8, stride=4)), nn.ReLU(),
            # init_(nn.Conv2d(32, 64, kernel_size=4, stride=2)), nn.ReLU(),
            # init_(nn.Conv2d(32, 64, kernel_size=4, stride=2)), nn.ReLU(),
            # init_(nn.Conv2d(64, 32, kernel_size=3, stride=1)),
            activation,
            Flatten(),
            # init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())
            init_(nn.Linear(hidden_size * h * w, hidden_size)),
            activation,
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, hidden_size, num_layers, recurrent, activation):
        assert num_layers > 0
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, init_normc_, lambda x: nn.init.constant_(x, 0))

        self.actor = nn.Sequential()
        self.critic = nn.Sequential()
        for i in range(num_layers):
            self.actor.add_module(
                name=f"fc{i}",
                module=nn.Sequential(
                    init_(nn.Linear(num_inputs, hidden_size)), activation
                ),
            )
            self.critic.add_module(
                name=f"fc{i}",
                module=nn.Sequential(
                    init_(nn.Linear(num_inputs, hidden_size)), activation
                ),
            )
            num_inputs = hidden_size

        self.critic_linear = init_(nn.Linear(num_inputs, 1))

        self.train()

    @property
    def output_size(self):
        return super().output_size

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
