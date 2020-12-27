# third party
import argparse
import csv
import random
import re
import subprocess
from dataclasses import fields, is_dataclass
from functools import reduce
from io import StringIO
from typing import List, Optional

import numpy as np
import torch
import torch.jit
import torch.nn as nn
from gym import spaces


def round(x, dec):
    return torch.round(x * 10 ** dec) / 10 ** dec


def grad(x, y):
    return torch.autograd.grad(
        x.mean(), y.parameters() if isinstance(y, nn.Module) else y, retain_graph=True
    )


def get_render_func(venv):
    if hasattr(venv, "envs"):
        return venv.envs[0].render
    elif hasattr(venv, "venv"):
        return get_render_func(venv.venv)
    elif hasattr(venv, "env"):
        return get_render_func(venv.env)

    return None


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


def set_index(array, idxs, value):
    idxs = np.array(idxs)
    if idxs.size > 0:
        array[tuple(idxs.T)] = value


def get_index(array, idxs):
    idxs = np.array(idxs)
    if idxs.size == 0:
        return np.array([], array.dtype)
    return array[tuple(idxs.T)]


def get_n_gpu():
    nvidia_smi = subprocess.check_output(
        "nvidia-smi --format=csv --query-gpu=memory.free".split(),
        universal_newlines=True,
    )
    return len(list(csv.reader(StringIO(nvidia_smi)))) - 1


def get_random_gpu():
    return random.randrange(0, get_n_gpu())


def get_freer_gpu():
    nvidia_smi = subprocess.check_output(
        "nvidia-smi --format=csv --query-gpu=memory.free".split(),
        universal_newlines=True,
    )
    free_memory = [
        float(x[0].split()[0])
        for i, x in enumerate(csv.reader(StringIO(nvidia_smi)))
        if i > 0
    ]
    return int(np.argmax(free_memory))


def init_(network: nn.Module, non_linearity: nn.Module = nn.ReLU):
    if non_linearity is None:
        return init(network, init_normc_, lambda x: nn.init.constant_(x, 0))
        # return init(network, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
    return init(
        network,
        nn.init.orthogonal_,
        lambda x: nn.init.constant_(x, 0),
        nn.init.calculate_gain(non_linearity.__name__.lower()),
    )


def broadcast3d(inputs, shape):
    return inputs.view(*inputs.shape, 1, 1).expand(*inputs.shape, *shape)


def interp(x1, x2, c):
    return c * x2 + (1 - c) * x1


@torch.jit.script
def log_prob(i, probs):
    return torch.log(torch.gather(probs, -1, i))


def trace(module_fn, in_size):
    return torch.jit.trace(module_fn(in_size), example_inputs=torch.rand(1, in_size))


RESET = "\033[0m"


def k_scalar_pairs(*args, **kwargs):
    for k, v in dict(*args, **kwargs).items():
        mean = np.mean(v)
        if not np.isnan(mean):
            yield k, mean


def set_seeds(cuda, cuda_deterministic, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cuda &= torch.cuda.is_available()
    if cuda and cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)


def hierarchical_parse_args(parser: argparse.ArgumentParser, include_positional=False):
    """
    :return:
    {
        group1: {**kwargs}
        group2: {**kwargs}
        ...
        **kwargs
    }
    """
    args = parser.parse_args()

    def key_value_pairs(group):
        for action in group._group_actions:
            if action.dest != "help":
                yield action.dest, getattr(args, action.dest, None)

    def get_positionals(groups):
        for group in groups:
            if group.title == "positional arguments":
                for k, v in key_value_pairs(group):
                    yield v

    def get_nonpositionals(groups: List[argparse._ArgumentGroup]):
        for group in groups:
            if group.title != "positional arguments":
                children = key_value_pairs(group)
                descendants = get_nonpositionals(group._action_groups)
                yield group.title, {**dict(children), **dict(descendants)}

    positional = list(get_positionals(parser._action_groups))
    nonpositional = dict(get_nonpositionals(parser._action_groups))
    optional = nonpositional.pop("optional arguments")
    nonpositional = {**nonpositional, **optional}
    if include_positional:
        return positional, nonpositional
    return nonpositional


def get_device(name):
    match = re.search("\d+$", name)
    if match:
        device_num = int(match.group()) % get_n_gpu()
    else:
        device_num = get_random_gpu()

    return torch.device("cuda", device_num)


def astuple(obj):
    def gen():
        for f in fields(obj):
            yield astuple(getattr(obj, f.name))

    if is_dataclass(obj):
        return tuple(gen())
    return obj


def asdict(obj):
    def gen():
        for f in fields(obj):
            yield f.name, asdict(getattr(obj, f.name))

    if hasattr(obj, "_asdict"):
        # noinspection PyProtectedMember
        return obj._asdict()
    if is_dataclass(obj):
        return dict(gen())
    return obj


class Discrete(spaces.Discrete):
    def __init__(self, low: int, high: int):
        self.low = low
        self.high = high
        super().__init__(1 + high - low)

    def sample(self) -> int:
        return self.low + super().sample()

    def contains(self, x) -> bool:
        return super().contains(x - self.low)

    def __repr__(self) -> str:
        return f"Discrete({self.low}, {self.high})"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Discrete)
            and self.low == other.low
            and self.high == other.high
        )


def get_max_shape(*xs) -> np.ndarray:
    def compare_shape(max_so_far: Optional[np.ndarray], opener: np.ndarray):
        new = np.array(opener.shape)
        return new if max_so_far is None else np.maximum(new, max_so_far)

    return reduce(compare_shape, map(np.array, xs), None)
