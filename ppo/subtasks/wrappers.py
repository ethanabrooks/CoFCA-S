from collections import namedtuple

import gym
from gym import spaces
from gym.spaces import Discrete
import numpy as np

Actions = namedtuple('Actions', 'a cr cg g')


class DebugWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.guess = 0
        self.truth = 0
        self.last_reward = None
        action_spaces = Actions(**env.action_space.spaces)
        for x in action_spaces:
            assert isinstance(x, Discrete)
        self.action_sections = len(action_spaces)
        self.truth = None

    def step(self, action):
        actions = Actions(*[x.item() for x in np.split(action, self.action_sections)])
        self.truth = int(self.env.unwrapped.subtask_idx)
        self.guess = int(actions.g)
        # r = -self.guess
        self.time_steps += 1
        # print("truth", truth)
        # print("guess", guess)
        # r = 0
        # if self.env.unwrapped.subtask is not None and self.guess != self.truth:
        # r = -0.1
        r = -self.guess
        s, _, t, i = super().step(action)
        self.last_reward = r
        return s, r, t, i

    def reset(self):
        self.time_steps = 0
        return super().reset()

    def render(self, mode="human"):
        print("########################################")
        super().render(sleep_time=0)
        print("guess", self.guess)
        print("truth", self.truth)
        print("reward", self.last_reward)
        input("pause")


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
            self.render_assigned_subtask()
        # input("paused")

    def render_assigned_subtask(self):
        env = self.env.unwrapped
        g_type, g_count, g_obj = tuple(env.subtasks[self.last_g])
        print(
            "Assigned subtask:",
            self.last_g,
            env.interactions[g_type],
            g_count + 1,
            env.object_types[g_obj],
        )
