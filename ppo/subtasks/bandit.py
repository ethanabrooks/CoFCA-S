from collections import namedtuple

from gym import spaces
from rl_utils import hierarchical_parse_args
import numpy as np
from ppo.subtasks import control_flow, keyboard_control
from ppo.subtasks.lines import If, Else, EndIf, While, EndWhile, Subtask

Obs = namedtuple("Obs", "active condition lines")
# TODO: this is very hacky. things will break unless namedtuple is in alphabetical order.


class Env(control_flow.Env):
    def __init__(self, seed, n_lines, flip_prob, time_limit):
        super().__init__(seed, n_lines)
        self.time_limit = time_limit
        self.flip_prob = flip_prob
        self.condition_bit = None
        self.last_action = None
        self.last_active = None
        self.last_reward = None
        self.line_types = [If, Else, EndIf, While, EndWhile, Subtask]
        self.action_space = spaces.Discrete(n_lines)
        self.observation_space = spaces.Dict(
            dict(
                condition=spaces.Discrete(2),
                lines=spaces.MultiDiscrete(np.array([len(self.line_types)] * n_lines)),
                active=spaces.Discrete(n_lines + 1),
            )
        )
        self.t = None

    def reset(self):
        self.last_action = None
        self.last_active = None
        self.last_reward = None
        self.t = 0
        self.condition_bit = self.random.randint(0, 2)
        return super().reset()

    def step(self, action):
        self.t += 1
        if self.time_limit and self.t > self.time_limit:
            return self.get_observation(), -1, True, {}
        if action == self.n_lines:
            # no-op
            return self.get_observation(), 0, False, {}
        self.last_action = action
        self.last_active = self.active
        if action != self.active:
            return self.get_observation(), -1, True, {}
        self.condition_bit = 1 - int(self.random.rand() < self.flip_prob)
        s, r, t, i = super().step(action)
        self.last_reward = r
        return s, r, t, i

    def get_observation(self):
        lines = [self.line_types.index(t) for t in self.lines]
        o = Obs(
            condition=self.condition_bit,
            lines=lines,
            active=self.n_lines if self.active is None else self.active,
        )._asdict()
        assert self.observation_space.contains(o)
        return o

    def _evaluate_condition(self, i=None):
        return bool(self.condition_bit)

    def done(self):
        return True  # bandits are always done

    def line_strings(self, index, level):
        if index == len(self.lines):
            return
        line = self.lines[index]
        if line in [Else, EndIf, EndWhile]:
            level -= 1
        if index == self.last_active and index == self.last_action:
            pre = "+ "
        elif index == self.last_action:
            pre = "- "
        elif index == self.last_active:
            pre = "| "
        else:
            pre = "  "
        indent = pre * level
        yield indent + line.__name__
        if line in [If, While, Else]:
            level += 1
        yield from self.line_strings(index + 1, level)

    def render(self, mode="human"):
        for i, string in enumerate(self.line_strings(index=0, level=1)):
            print(f"{i}{string}")
        print("Condition:", self.condition_bit)
        print("Reward:", self.last_reward)
        input("pause")

    def train(self):
        print("No logic is currently implemented for the train() method")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--n-lines", default=6, type=int)
    parser.add_argument("--flip-prob", default=0.5, type=float)
    args = hierarchical_parse_args(parser)
    keyboard_control.run(Env(**args), actions="".join(map(str, range(args["n_lines"]))))
