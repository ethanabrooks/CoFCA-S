from collections import OrderedDict, namedtuple, Counter

import numpy as np
from gym import spaces

import upper_env
import keyboard_control
from upper_env import Action
from enums import Interaction, Resource, Other, Terrain, InventoryItems

Obs = namedtuple("Obs", "inventory line obs")
action_space = spaces.Discrete(len(list(upper_env.lower_level_actions())))


class Env(upper_env.Env):
    def __init__(self, *args, **kwargs):
        kwargs.update(
            bridge_failure_prob=0,
            bandit_prob=0,
            windfall_prob=0,
            map_discovery_prob=0,
        )
        super().__init__(*args, **kwargs)
        self.observation_space = self.observation_space_from_upper(
            self.observation_space
        )
        self.action_space = action_space

    @staticmethod
    def observation_space_from_upper(observation_space):
        line_space = observation_space.spaces["lines"].nvec[0]
        return spaces.Dict(
            dict(
                {k: v for k, v in observation_space.spaces.items() if k in Obs._fields},
                line=spaces.MultiDiscrete(line_space + 1),  # + 1 for cross mountain
            )
        )

    def step(self, action: int):
        action = Action(
            upper=0, lower=self.lower_level_actions[action], delta=0, dg=0, ptr=0
        )
        return self.iterator.send(action)

    def initialize_inventory(self):
        inventory = super().initialize_inventory()
        if self.random.random() < 1 / len(self.subtasks):
            inventory.add(Other.MAP)

            # random inventory
            prob = 1 / len(InventoryItems)
            for item in InventoryItems:
                if self.random.random() < prob:
                    inventory.add(item)
        return inventory

    def state_generator(self, *blocks):
        iterator = super().state_generator(*blocks)
        action = None
        line, *_ = self.get_lines(*blocks)

        def check_success(success, subtasks_completed, **_):
            return success or line in subtasks_completed

        while True:
            state, render = iterator.send(action)
            state.update(success=check_success(**state), line=line)
            action = yield state, render

    def obs_generator(self, *lines):
        iterator = super().obs_generator(*lines)
        state = yield next(iterator)

        def build_obs(inventory, obs, **_):
            return Obs(
                inventory=inventory,
                line=(
                    self.line_space if cross_mountain else self.preprocess_line(line)
                ),
                obs=obs,
            )

        while True:
            cross_mountain = state["should_cross_mountain"]
            line = state["line"]
            _obs, _render = iterator.send(state)
            _obs = OrderedDict(build_obs(**_obs)._asdict())

            def render():
                _render()
                print(
                    "Line:",
                    "Cross Mountain" if cross_mountain else str(line),
                )

            state = yield _obs, render

    # noinspection PyMethodOverriding
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
    Env.add_arguments(PARSER)
    PARSER.add_argument("--seed", default=0, type=int)
    Env(rank=0, min_eval_lines=0, max_eval_lines=10, **vars(PARSER.parse_args())).main()
