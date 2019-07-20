import gym
import numpy as np

import ppo.control_flow
from ppo.control_flow import Actions


class DebugWrapper(ppo.control_flow.DebugWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.completed_subtask = False

    def reset(self):
        self.completed_subtask = False
        return super().reset()

    def step(self, action):
        actions = Actions(*[x.item() for x in np.split(action, self.action_sections)])
        env = self.env.unwrapped
        self.truth = int(env.subtask_idx)

        def lines_to_subtasks():
            i = 0
            for line in env.lines:
                if line[-1] == 0:
                    yield i
                    i += 1
                else:
                    yield None

        line = self.guess = int(actions.g)
        subtask = list(lines_to_subtasks())[line]
        r = 0

        if (subtask is not None and subtask != self.truth) or (
            subtask is None and not self.completed_subtask
        ):  # wrong subtask line
            # import ipdb

            # ipdb.set_trace()
            r = -0.1
        subtask_before = env.subtask_idx
        s, _, t, i = gym.Wrapper.step(self, action)
        subtask_after = env.subtask_idx
        self.completed_subtask = subtask_before != subtask_after
        self.last_reward = r
        return s, r, t, i


class Wrapper(ppo.control_flow.Wrapper):
    def render_assigned_subtask(self):
        try:
            print(f"{self.last_g}:{self.env.unwrapped.subtasks[self.last_g]}")
        except IndexError:
            return
