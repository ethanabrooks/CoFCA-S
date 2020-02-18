from collections import defaultdict

from gym.spaces import Discrete, MultiDiscrete

import ppo.control_flow.multi_step.env
from ppo.control_flow.env import State
from ppo.control_flow.lines import If, While
import numpy as np


class Env(ppo.control_flow.multi_step.env.Env):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.line = None
        self.choice = None
        self.action_space = MultiDiscrete(2 * np.ones(2))

    def state_generator(self, lines):
        object_pos = self.populate_world(lines)
        o = self.objects[int(self.random.choice(len(self.objects)))]
        self.line = line = (If, While)[int(self.random.choice(2))](id=o)
        agent_pos = self.random.randint(0, self.world_size, size=2)

        condition_evaluations = defaultdict(list)
        self.choice = None
        self.choice = yield State(
            obs=self.world_array(object_pos, agent_pos),
            condition=None,
            prev=0,
            ptr=0,
            condition_evaluations=condition_evaluations,
            term=False,
        )
        evaluation = self.evaluate_line(line, object_pos, condition_evaluations)
        _ = yield State(
            obs=self.world_array(object_pos, agent_pos),
            condition=None,
            prev=0,
            ptr=None if self.choice == evaluation else 0,
            condition_evaluations=condition_evaluations,
            term=True,
        )
        raise RuntimeError

    def render(self, mode="human", pause=True):
        self._render()
        print("line:", self.line)
        print("choice:", self.choice)
        if pause:
            input("pause")
