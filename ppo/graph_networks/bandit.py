from collections import namedtuple

from gym import spaces
from rl_utils import hierarchical_parse_args
import numpy as np
from ppo.graph_networks import control_flow, keyboard_control
from ppo.graph_networks.lines import If, Else, EndIf, While, EndWhile, Subtask, Padding

Obs = namedtuple("Obs", "condition lines")
# TODO: this is very hacky. things will break unless namedtuple is in alphabetical order.


class Env(control_flow.Env):
    def __init__(self, flip_prob, eval_lines, time_limit, baseline, **kwargs):
        super().__init__(eval_lines=eval_lines, **kwargs)
        self.time_limit = time_limit
        self.flip_prob = flip_prob
        self.baseline = baseline
        self.condition_bit = None
        self.last_action = None
        self.last_active = None
        self.last_reward = None
        self.line_types = [If, Else, EndIf, While, EndWhile, Subtask, Padding]
        self.action_space = spaces.Discrete(eval_lines)
        if baseline:
            self.observation_space = spaces.MultiBinary(
                2 + len(self.line_types) * eval_lines
            )
            self.eye = Obs(condition=np.eye(2), lines=np.eye(len(self.line_types)))
        else:
            self.observation_space = spaces.Dict(
                dict(
                    condition=spaces.Discrete(2),
                    lines=spaces.MultiDiscrete(
                        np.array([len(self.line_types)] * eval_lines)
                    ),
                    # active=spaces.Discrete(n_lines + 1),
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
        if action == len(self.lines):
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
        padded = self.lines + [Padding] * (self.eval_lines - len(self.lines))
        lines = [self.line_types.index(t) for t in padded]
        obs = Obs(
            condition=self.condition_bit,
            lines=lines,
            # active=self.n_lines if self.active is None else self.active,
        )
        if self.baseline:
            obs = [eye[o].flatten() for eye, o in zip(self.eye, obs)]
            obs = np.concatenate(obs)
        else:
            obs = obs._asdict()
        assert self.observation_space.contains(obs)
        return obs

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--n-lines", default=6, type=int)
    parser.add_argument("--flip-prob", default=0.5, type=float)
    args = hierarchical_parse_args(parser)
    keyboard_control.run(Env(**args), actions="".join(map(str, range(args["n_lines"]))))
