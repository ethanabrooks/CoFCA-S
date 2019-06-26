from collections import namedtuple

import gym
from gym import spaces
from gym.spaces import Box
import numpy as np

import gridworld_env.control_flow_gridworld
import ppo.subtasks.wrappers

Obs = namedtuple('Obs',
                 'base subtask subtasks conditions control next_subtask')


class Wrapper(ppo.subtasks.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        obs_spaces = gridworld_env.control_flow_gridworld.Obs(
            *env.observation_space.spaces)
        obs_spaces = Obs(
            base=obs_spaces.base,
            subtasks=obs_spaces.subtasks,
            conditions=obs_spaces.conditions,
            control=obs_spaces.control,
            subtask=spaces.Discrete(obs_spaces.subtasks.nvec.shape[0]),
            next_subtask=spaces.Discrete(2),
        )
        # noinspection PyProtectedMember
        self.observation_space = spaces.Tuple(
            obs_spaces._replace(base=Box(0, 1, shape=obs_spaces.base.nvec), ))
        self.action_space = spaces.Tuple(
            ppo.subtasks.Actions(
                a=env.action_space,
                g=spaces.Discrete(env.n_subtasks + 1),
                cg=spaces.Discrete(2),
                cr=spaces.Discrete(2),
            ))
        self.last_g = None

    def wrap_observation(self, observation):
        obs = gridworld_env.control_flow_gridworld.Obs(*observation)
        obs = Obs(
            base=obs.base,
            subtasks=obs.subtasks,
            conditions=obs.conditions,
            control=obs.control,
            subtask=[self.env.unwrapped.subtask_idx],
            next_subtask=[self.env.unwrapped.next_subtask])
        # print([np.shape(x) for x in obs])
        return np.concatenate([np.array(list(x)).flatten() for x in obs])
