# third party
import torch
import torch.nn as nn

# first party
from ppo.envs import VecNormalize


# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


def mlp(num_inputs, hidden_size, num_layers, activation, name='fc', num_outputs=None):
    init_ = lambda m: init(m, weight_init=init_normc_,
                           bias_init=lambda x: nn.init.constant_(x, 0))
    network = nn.Sequential()
    in_features = num_inputs
    for i in range(num_layers):
        network.add_module(name=f'{name}{i}',
                           module=nn.Sequential(
                               init_(nn.Linear(in_features, hidden_size)),
                               activation,
                           ))
        in_features = hidden_size
    if num_outputs:
        network.add_module(name=f'{name}-out',
                           module=init_(nn.Linear(in_features, num_outputs)))
    return network


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))
