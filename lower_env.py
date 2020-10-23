from collections import OrderedDict, namedtuple, Counter

import numpy as np
from gym import spaces

import upper_env
import keyboard_control
from upper_env import Action, CrossMountain
from enums import Interaction, Resource, Other, Terrain, InventoryItems


Obs = namedtuple("Obs", "inventory line obs")


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
        self.action_space = spaces.Discrete(len(list(self.lower_level_actions)))

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
            upper=self.subtasks[0],
            lower=self.lower_level_actions[action],
            delta=0,
            dg=0,
            ptr=0,
        )
        return self.iterator.send(action)

    def initialize_inventory(self):
        inventory = super().initialize_inventory()
        # random inventory
        prob = 1 / len(InventoryItems)
        for item in InventoryItems:
            if self.random.random() < prob:
                inventory.add(item)
        return inventory

    def time_limit(self, lines):
        return self.time_per_subtask()

    def state_generator(self, line, *lines):
        iterator = super().state_generator(line, *lines)
        state, render = next(iterator)

        if Other.MAP in state["inventory"]:
            line = CrossMountain

        def check_success(success, subtasks_completed, **_):
            return success or line in subtasks_completed

        while True:
            state.update(success=check_success(**state), line=line)
            action = yield state, render
            state, render = iterator.send(action)

    def obs_generator(self, *lines):
        iterator = super().obs_generator(*lines)
        state = yield next(iterator)

        def build_obs(inventory, obs, **_):
            return Obs(
                inventory=inventory,
                line=self.preprocess_line(line),
                obs=obs,
            )

        while True:
            line = state["line"]
            _obs, _render = iterator.send(state)
            _obs = OrderedDict(build_obs(**_obs)._asdict())

            def render():
                _render()
                print("Line:", str(line))

            state = yield _obs, render

    # noinspection PyMethodOverriding
    def main(self):
        actions = [
            tuple(x) if isinstance(x, np.ndarray) else x
            for x in self.lower_level_actions
        ]
        mapping = dict(
            # w=(-1, 0),
            # s=(1, 0),
            # a=(0, -1),
            # d=(0, 1),
            c=Interaction.COLLECT,
            r=Interaction.REFINE,
        )

        def action_fn(string):
            try:
                action = tuple(map(int, string.split()))
            except ValueError:
                action = mapping.get(string, None)
            if action is None:
                return None
            if action in actions:
                return actions.index(action)

        keyboard_control.run(self, action_fn=action_fn)


def main(debug_env, **kwargs):
    Env(rank=0, min_eval_lines=0, max_eval_lines=10, eval_steps=0, **kwargs).main()


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser()
    Env.add_arguments(PARSER)
    PARSER.add_argument("--seed", default=0, type=int)
    main(**vars(PARSER.parse_args()))
