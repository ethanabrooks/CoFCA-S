from collections import defaultdict, Counter

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
        object_pos, lines = self.populate_world(lines)
        passing = self.random.choice(2)
        o1, o2 = self.random.choice(len(self.objects), size=2)
        o1, o2 = self.objects[o1], self.objects[o2]
        line_type = (If, While)[int(self.random.choice(2))]
        count1 = sum(1 for o, _ in object_pos if o == o1)
        count2 = sum(1 for o, _ in object_pos if o == o2)
        self.line = line = line_type(
            (o1, o2) if (passing and count1 < count2) else (o2, o1)
        )
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

    def get_observation(self, obs, active, lines):
        return super().get_observation(obs, active, [self.line])

    def render(self, mode="human", pause=True):
        self._render()
        o1, o2 = self.line.id
        print(f"line: {self.line}: count[{o1}] < count[{o2}]")
        print("choice:", self.choice)
        if pause:
            input("pause")
