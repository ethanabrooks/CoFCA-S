from collections import namedtuple

import gym
import numpy as np
from gym import spaces
from gym.spaces import Box, Discrete

Actions = namedtuple('Actions', 'a cr cg g')
Obs = namedtuple('Obs', 'base subtask subtasks next_subtask')


class DebugWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_guess = None
        self.last_reward = None
        action_spaces = Actions(*env.action_space.spaces)
        for x in action_spaces:
            assert isinstance(x, Discrete)
        self.action_sections = len(action_spaces)
        self.truth = None

    def step(self, action):
        actions = Actions(*[x.item() for x in np.split(action, self.action_sections)])
        s, _, t, i = super().step(action)
        guess = int(actions.g)
        env = self.env.unwrapped
        truth = int(env.subtask_idx)
        if truth > env.n_subtasks:  # truth is out of bounds
            truth = self.truth  # keep truth at old value

        r = float(np.all(guess == truth)) - 1
        # if r < 0:
        #     import ipdb
        #     ipdb.set_trace()

        self.truth = truth
        self.last_guess = guess
        self.last_reward = r
        return s, r, t, i

    def render(self, mode='human'):
        print('########################################')
        super().render(sleep_time=0)
        print('guess', self.last_guess)
        print('truth', self.env.unwrapped.subtask_idx)
        print('$$$$$$$$$$$$$$')
        print('$ reward', self.last_reward, '$')
        print('$$$$$$$$$$$$$$')
        # input('pause')


class Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_space, subtasks_space = env.observation_space.spaces
        assert np.all(subtasks_space.nvec == subtasks_space.nvec[0])
        self.subtasks_space = subtasks_space
        self.observation_space = spaces.Tuple(
            Obs(
                base=Box(0, 1, shape=obs_space.nvec),
                subtask=spaces.Discrete(subtasks_space.nvec.shape[0]),
                subtasks=subtasks_space,
                next_subtask=spaces.Discrete(2),
            ))
        self.action_space = spaces.Tuple(
            Actions(
                a=env.action_space,
                g=spaces.Discrete(env.n_subtasks),
                cg=spaces.Discrete(2),
                cr=spaces.Discrete(2),
            ))
        self.last_g = None

    def step(self, action):
        actions = Actions(*np.split(action, len(self.action_space.spaces)))
        action = int(actions.a)
        self.last_g = int(actions.g)
        s, r, t, i = super().step(action)
        return self.wrap_observation(s), r, t, i

    def reset(self, **kwargs):
        return self.wrap_observation(super().reset())

    def wrap_observation(self, observation):
        obs, *_ = observation
        _, h, w = obs.shape
        env = self.env.unwrapped
        observation = Obs(
            base=obs,
            subtask=env.subtask_idx,
            subtasks=env.subtasks,
            next_subtask=env.next_subtask)
        # for obs, space in zip(observation, self.observation_space.spaces):
        # assert space.contains(np.array(obs))
        return np.concatenate([np.array(x).flatten() for x in observation])

    def render(self, mode='human', **kwargs):
        super().render(mode=mode)
        if self.last_g is not None:
            env = self.env.unwrapped
            subtask = env.subtasks[self.last_g]
            print('Assigned subtask:', f'{self.last_g}:{subtask}')
        input('paused')
