from collections import namedtuple

import gym
from gym import spaces
from gym.spaces import Discrete
import numpy as np

Actions = namedtuple('Actions', 'a cr cg g')


class DebugWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_guess = None
        self.last_reward = None
        action_spaces = Actions(**env.action_space.spaces)
        for x in action_spaces:
            assert isinstance(x, Discrete)
        self.action_sections = len(action_spaces)
        self.truth = None

    def step(self, action):
        actions = Actions(*[x.item() for x in np.split(action, self.action_sections)])
        truth = int(self.env.unwrapped.subtask_idx)
        guess = int(actions.g)
        r = 0
        if self.env.unwrapped.subtask is not None and guess != truth:
            r = -1
        r = -np.abs(self.env.unwrapped.next_subtask - actions.cr)
        s, _, t, i = super().step(action)
        self.last_guess = guess
        self.last_reward = r
        return s, r, t, i

    def render(self, mode="human"):
        print("########################################")
        super().render(sleep_time=0)
        # print("guess", self.last_guess)
        # print("truth", self.env.unwrapped.subtask_idx)
        print("reward", self.last_reward)
        # input('pause')


class Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Dict(
            Actions(
                a=env.action_space,
                g=spaces.Discrete(env.n_subtasks),
                cg=spaces.Discrete(2),
                cr=spaces.Discrete(2),
            )._asdict())
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

    def render(self, mode="human", **kwargs):
        super().render(mode=mode)
        if self.last_g is not None:
            env = self.env.unwrapped
            g_type, g_count, g_obj = tuple(env.subtasks[self.last_g])
            print(
                "Assigned subtask:",
                self.last_g,
                env.interactions[g_type],
                g_count + 1,
                env.object_types[g_obj],
            )
        input("paused")
