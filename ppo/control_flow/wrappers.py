from collections import namedtuple

import gym
from gym import spaces
from gym.spaces import Discrete
import numpy as np

from common.vec_env.util import space_shape
from gridworld_env.control_flow_gridworld import LineTypes, TaskTypes
from ppo.utils import RED, RESET

Actions = namedtuple("Actions", "a cr cg g z")


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

    def step(self, action):
        actions = Actions(*[x.item() for x in np.split(action, self.action_sections)])
        self.truth = int(self.env.unwrapped.subtask_idx)
        self.guess = int(actions.g)
        # print("truth", truth)
        # print("guess", guess)
        r = 0
        if self.env.unwrapped.subtask is not None and self.guess != self.truth:
            r = -0.1
            import ipdb

            ipdb.set_trace()
        s, _, t, i = super().step(action)
        self.last_reward = r
        return s, r, t, i

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
        self.action_space = spaces.Dict(
            Actions(
                a=env.action_space,
                g=spaces.Discrete(env.unwrapped.n_subtasks),
                cg=spaces.Discrete(2),
                cr=spaces.Discrete(2),
                z=spaces.MultiDiscrete(
                    np.full(env.unwrapped.n_subtasks, len(LineTypes._fields))
                ),
            )._asdict()
        )
        self.action_sections = np.cumsum(
            [s for s, in space_shape(self.action_space).values()]
        )[:-1]
        self.last_g = None

        self.auto_curriculum = self.env.unwrapped.task_type is TaskTypes.Auto
        self.task_types = (t for t in TaskTypes if t is not TaskTypes.Auto)
        self.task_type = None

    def step(self, action):
        actions = Actions(*np.split(action, self.action_sections))
        action = int(actions.a)
        self.last_g = int(actions.g)
        s, r, t, i = super().step(action)
        if r == 1:
            self.consecutive_successes += 1
        elif t and r < 0:
            self.consecutive_successes = 0
        i.update(
            task_type_idx=self.env.unwrapped.task_type.value,
            consecutive_successes=self.consecutive_successes,
        )
        return s, r, t, i

    def reset(self):
        if self.auto_curriculum and (
            self.task_type is None or self.consecutive_successes >= 500
        ):
            self.task_type = next(self.task_types, TaskTypes.General)
            self.env.unwrapped.task_type = self.task_type
        return super().reset()

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
