import gym
from gym import spaces
import numpy as np

import ppo.subtasks
from ppo.subtasks import Actions


class DebugWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_sections = len(env.action_space.spaces)

    def step(self, action):
        actions = Actions(*[x.item() for x in np.split(action, self.action_sections)])

        def lines_to_subtask():
            i = 0
            for line in self.env.unwrapped.lines:
                if line[-1] > 0:
                    yield i + 1
                else:
                    yield i
                i += 1

        self.truth = int(self.env.unwrapped.subtask_idx)
        self.guess = list(lines_to_subtask())[int(actions.g)]
        print("debug_wrapper lines_to_subtask", list(lines_to_subtask()))
        print("debug_wrapper subtasks", self.env.unwrapped.subtasks)
        print("debug_wrapper lines", self.env.unwrapped.lines)
        # r = -self.guess
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


class Wrapper(ppo.subtasks.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space.spaces.update(
            g=spaces.Discrete(len(env.observation_space.spaces["lines"].nvec))
        )

    def render_assigned_subtask(self):
        env = self.env.unwrapped
        g_type, g_count, g_obj, condition = tuple(env.lines[self.last_g])
        if condition:
            print("if", env.object_types[condition - 1])
        else:
            print(
                self.last_g,
                env.interactions[g_type],
                g_count + 1,
                env.object_types[g_obj],
            )
