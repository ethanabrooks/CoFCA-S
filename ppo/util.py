# third party
import torch
import torch.nn as nn

# first party
from ppo.envs import VecNormalize

ACTIVATIONS = dict(
    relu=nn.ReLU,
    leaky=nn.LeakyReLU,
    elu=nn.ELU,
    selu=nn.SELU,
    prelu=nn.PReLU,
    sigmoid=nn.Sigmoid,
    tanh=nn.Tanh,
    none=None,
)


def parse_activation(arg: str):
    return ACTIVATIONS[arg]()


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


def mlp(num_inputs,
        hidden_size,
        num_layers,
        activation,
        name='fc',
        num_outputs=None):
    init_ = lambda m: init(m, weight_init=init_normc_,
                           bias_init=lambda x: nn.init.constant_(x, 0),
                           gain=.1)
    network = nn.Sequential()
    in_features = num_inputs
    for i in range(num_layers):
        network.add_module(
            name=f'{name}{i}',
            module=nn.Sequential(
                init_(nn.Linear(in_features, hidden_size)),
                activation,
            ))
        in_features = hidden_size
    if num_outputs:
        network.add_module(
            name=f'{name}-out',
            module=init_(nn.Linear(in_features, num_outputs)))
    return network


class Categorical(torch.distributions.Categorical):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d = self.probs.numel()
        self.range = torch.arange(0., float(d))
        if len(self.probs.size()) == 2:
            n, _ = self.probs.size()
            self.range = self.range.repeat(n, 1)

    @property
    def mean(self):
        return torch.sum(self.range * self.probs, dim=-1)

    @property
    def variance(self):
        return torch.sum(
            self.probs * ((self.range - self.mean.unsqueeze(-1))**2), dim=-1)


class NoInput(nn.Module):
    def __init__(self, size):
        super().__init__()
        tensor = torch.Tensor(1, size)
        init_normc_(tensor)
        self.weight = nn.Parameter(tensor)

    def forward(self, inputs):
        size, *_ = inputs.size()
        return self.weight.repeat(size, 1)


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
