from collections import OrderedDict, namedtuple

import numpy as np
from colored import fg
from gym import spaces

import keyboard_control
import upper_env
from objects import Interaction, Resource
from upper_env import Action

Obs = namedtuple("Obs", "inventory line obs")


class Env(upper_env.Env):
    def __init__(self, *args, **kwargs):
        kwargs.update(
            bridge_failure_prob=0,
            map_discovery_prob=0,
            bandit_prob=0,
            windfall_prob=0,
        )
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
        return self.iterator.send(action)

    def generator(self):
        iterator = super().generator()
        action = None
        time_limit = self.time_per_subtask()
        while True:
            s, r, t, i = iterator.send(action)
            time_limit -= 1
            if time_limit == 0:
                t = True
            action = yield s, r, t, i

    def preprocess_state(
        self, room, lines, inventory, info, done, ptr, subtask_complete, **kwargs
    ):
        ptr = min(ptr, len(lines) - 1)
        obs = OrderedDict(
            Obs(
                obs=room,
                line=self.preprocess_line(lines[ptr]),
                inventory=self.inventory_representation(inventory),
            )._asdict()
        )
        # for name, space in self.observation_space.spaces.items():
        #     if not space.contains(obs[name]):
        #         space.contains(obs[name])
        if done or subtask_complete:
            info.update(success=subtask_complete)

        reward = -0.1
        return obs, reward, subtask_complete, info

    def _render(self, subtask_complete, **kwargs):
        if subtask_complete:
            print(fg("green"))
        super()._render(**kwargs, subtask_complete=subtask_complete)

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
            return actions.index(action)

        keyboard_control.run(self, action_fn=action_fn)


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--min-eval-lines", type=int)
    PARSER.add_argument("--max-eval-lines", type=int)
    Env.add_arguments(PARSER)
    PARSER.add_argument("--seed", default=0, type=int)
    Env(rank=0, **vars(PARSER.parse_args())).main()
