"""
Helpers for dealing with vectorized environments.
"""

from collections import OrderedDict

import gym
import gym.spaces
import numpy as np


def copy_obs_dict(obs):
    """
    Deep-copy an observation dict.
    """
    return {k: np.copy(v) for k, v in obs.items()}


def dict_to_obs(obs_dict):
    """
    Convert an observation dict into a raw array if the
    original observation space was not a Dict space.
    """
    if set(obs_dict.keys()) == {None}:
        return obs_dict[None]
    return obs_dict


def space_shape(space: gym.Space):
    if isinstance(space, gym.spaces.Box):
        return space.low.shape
    if isinstance(space, gym.spaces.Dict):
        return {k: space_shape(v) for k, v in space.spaces.items()}
    if isinstance(space, gym.spaces.Tuple):
        return tuple(space_shape(s) for s in space.spaces)
    if isinstance(space, gym.spaces.MultiDiscrete):
        return space.nvec.shape
    if isinstance(space, gym.spaces.Discrete):
        return (1,)
    if isinstance(space, gym.spaces.MultiBinary):
        return (space.n,)
    raise NotImplementedError


def buffer_shape(space: gym.Space):
    shape = space_shape(space)
    if not all(isinstance(d, int) for d in shape):
        # print('buffer shape', shape)
        shape = (int(sum(np.prod(s) for s in shape)),)  # concatenate
    return shape


def obs_space_info(obs_space):
    """
    Get dict-structured information about a gym.Space.

    Returns:
      A tuple (keys, shapes, dtypes):
        keys: a list of dict keys.
        shapes: a dict mapping keys to shapes.
        dtypes: a dict mapping keys to dtypes.
    """
    if isinstance(obs_space, gym.spaces.Dict):
        assert isinstance(obs_space.spaces, OrderedDict)
        subspaces = obs_space.spaces
    else:
        subspaces = {None: obs_space}
    keys = []
    shapes = {}
    dtypes = {}
    for key, space in subspaces.items():
        keys.append(key)
        shape = buffer_shape(space)
        shapes[key] = shape
        dtypes[key] = space.dtype
    return keys, shapes, dtypes


def obs_to_dict(obs):
    """
    Convert an observation into a dict.
    """
    if isinstance(obs, dict):
        return obs
    return {None: obs}
