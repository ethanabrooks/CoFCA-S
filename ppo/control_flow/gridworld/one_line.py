from collections import defaultdict, Counter

from gym.spaces import Discrete, MultiDiscrete

import ppo.control_flow.gridworld.env
from ppo.control_flow.env import State
from ppo.control_flow.lines import If, While
import numpy as np


class Env(ppo.control_flow.gridworld.env.Env):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.line = None
        self.choice = None
        self.action_space = MultiDiscrete(2 * np.ones(2))

    def state_generator(self, lines):
        object_pos, lines = self.populate_world(lines)
        running_count = Counter()
        for o, _ in object_pos:
            running_count[o] += 1
        passing = self.random.choice(2)
        o = self.objects[int(self.random.choice(len(self.objects)))]
        line_type = (If, While)[int(self.random.choice(2))]
        if line_type is If:
            count_plus_1 = min(self.max_comparison_number, running_count[o] + 1)
            comparison_number = (
                self.random.randint(0, count_plus_1)
                if passing
                else self.random.randint(count_plus_1, self.max_comparison_number + 1)
            )
        else:
            assert line_type is While
            comparison_number = running_count[o] + 1
        self.line = line = line_type((comparison_number, o))
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
        print("line:", self.line)
        print("choice:", self.choice)
        if pause:
            input("pause")
