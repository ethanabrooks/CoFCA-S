from collections import namedtuple

import gym
import numpy as np
from gym import spaces

from common.vec_env.util import space_shape
from ppo.utils import RED, RESET

Actions = namedtuple("Actions", "a cr cg g z l")


class DebugWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.guess = 0
        self.truth = 0
        self.last_reward = None
        sections = [s for s, in space_shape(self.action_space).values()]
        self.action_sections = np.cumsum(sections)[:-1]

    def step(self, action: np.ndarray):
        actions = Actions(*np.split(action, self.action_sections))
        env = self.env.unwrapped
        # self.guess = actions.l
        if self.guess is None:
            # first step
            self.truth = env.last_condition_passed
            self.guess = actions.l
        # print("truth", self.truth)
        # print("guess", self.guess)
        # if self.env.unwrapped.subtask is not None and self.guess != self.truth:
        # r = -0.1
        s, r, t, i = super().step(action)
        if t:
            _r = 1 if bool(self.truth) == bool(self.guess) else -0.1
            i.update(matching=float(_r == r))
        self.last_reward = r
        return s, r, t, i

    def reset(self, **kwargs):
        self.guess = None
        self.truth = None
        return super().reset(**kwargs)

    def render(self, mode="human"):
        print("guess", self.guess)
        print("truth", self.truth)
        print("reward", self.last_reward)
        super().render(sleep_time=0)
        print("########################################")


class Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.consecutive_successes = 0

    def step(self, action):
        s, r, t, i = super().step(action)
        if r == 1:
            self.consecutive_successes += 1
        elif t and r < 0:
            self.consecutive_successes = 0
        i.update(consecutive_successes=self.consecutive_successes)
        return s, r, t, i

    def render(self, mode="human", **kwargs):
        if self.last_g is not None:
            self.render_assigned_subtask()
        super().render(mode=mode)
        if self.env._elapsed_steps == self.env._max_episode_steps:
            print(
                RED
                + "***********************************************************************************"
            )
            print(
                "                                   Task Failed                                   "
            )
            print(
                "***********************************************************************************"
                + RESET
            )
        input("paused")

    def render_assigned_subtask(self):
        env = self.env.unwrapped
        if self.last_g < len(env.subtasks):
            print("❯❯ Assigned subtask:", self.last_g, env.subtasks[self.last_g])
