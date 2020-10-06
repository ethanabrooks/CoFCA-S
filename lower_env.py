from collections import OrderedDict, namedtuple

import numpy as np
from gym import spaces

import keyboard_control
import upper_env
from upper_env import lower_level_actions, Action
from objects import Interaction, Resource

Obs = namedtuple("Obs", "inventory line obs")


class Env(upper_env.Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space.spaces["line"] = spaces.MultiDiscrete(
            np.array(self.line_space)
        )
        self.observation_space = spaces.Dict(
            {k: v for k, v in self.observation_space.spaces.items() if k in Obs._fields}
        )

        self.action_space = spaces.Discrete(len(self.lower_level_actions))

    def step(self, action: int):
        action = Action(
            upper=0, lower=self.lower_level_actions[action], delta=0, dg=0, ptr=0
        )
        return self.preprocess_state(**self.iterator.send(action))

    def preprocess_state(
        self,
        room,
        lines,
        inventory,
        inventory_change,
        done,
        info,
        ptr,
        subtask_complete,
    ):
        ptr = min(ptr, len(lines) - 1)
        obs = Obs(
            obs=room,
            line=self.preprocess_line(lines[ptr]),
            inventory=self.inventory_representation(inventory),
        )
        obs = OrderedDict(obs._asdict())
        # for name, space in self.observation_space.spaces.items():
        #     if not space.contains(obs[name]):
        #         space.contains(obs[name])
        reward = bool(subtask_complete)
        return obs, reward, done, info

    def main(self):
        actions = [
            tuple(x) if isinstance(x, np.ndarray) else x
            for x in self.lower_level_actions
        ]
        mapping = dict(
            w=(-1, 0),
            s=(1, 0),
            a=(0, -1),
            d=(0, 1),
            c=Interaction.COLLECT,
            t=Resource.STONE,
            i=Resource.IRON,
            o=Resource.WOOD,
        )

        def action_fn(string):
            action = mapping.get(string, None)
            if action is None:
                return None
            action = actions.index(action)
            return np.array(Action(upper=0, lower=action, delta=0, dg=0, ptr=0))

        keyboard_control.run(self, action_fn=action_fn)


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--min-eval-lines", type=int)
    PARSER.add_argument("--max-eval-lines", type=int)
    Env.add_arguments(PARSER)
    PARSER.add_argument("--seed", default=0, type=int)
    Env(rank=0, **vars(PARSER.parse_args())).main()
