from dataclasses import dataclass
from typing import Tuple

import env
import keyboard_control
from data_types import (
    ActionType,
    Building,
    X,
    WorkerID,
    Assignment,
    CompoundAction,
    State,
    Action,
    BuildOrder,
    Coord,
    WorkerAction,
    Movement,
    Targets,
    WORLD_SIZE,
)


@dataclass(frozen=True)
class DebugAction(Action):
    def next(self) -> ActionType:
        if self.reset():
            return DebugAction1
        return self.next_if_not_reset()


@dataclass(frozen=True)
class DebugAction1(DebugAction):
    building: X

    @classmethod
    def num_values(cls) -> "DebugAction1":
        return cls(len(Building))

    def reset(self):
        return True

    def next_if_not_reset(self) -> ActionType:
        raise NotImplementedError

    @classmethod
    def parse(cls, a) -> "Action":
        return cls(Building(1 + a))


@dataclass(frozen=True)
class DebugBuildOrder(BuildOrder):
    def action(self, current_position: Coord, *args, **kwargs) -> "WorkerAction":
        return self.building


@dataclass(frozen=True)
class DebugCompoundAction(CompoundAction):
    action1: DebugAction1 = None
    ptr: int = 0
    active: ActionType = DebugAction1

    @classmethod
    def classes(cls):
        yield DebugAction1

    def actions(self):
        yield self.action1

    def worker(self) -> WorkerID:
        return WorkerID(1)

    def assignment(self) -> Assignment:
        return DebugBuildOrder(self.action1.building, location=(0, 0))

    def is_op(self):
        return True


class Env(env.Env):
    @staticmethod
    def building_allowed(*args, **kwargs):
        return True

    @staticmethod
    def compound_action(*args, **kwargs) -> DebugCompoundAction:
        return DebugCompoundAction(*args, **kwargs)

    def done_generator(self, *lines):
        while True:
            yield True, lambda: None

    def main(self):
        def action_fn(string: str):
            try:
                return self.compound_action(
                    DebugAction1(building=Targets[int(string)]),
                )
            except (ValueError, TypeError) as e:
                print(e)

        keyboard_control.run(self, action_fn)

    def reward_generator(self):
        state: State
        state = yield
        while True:
            reward = int(state.success)

            # noinspection PyTypeChecker
            state = yield reward, lambda: print("Reward:", reward)

    @staticmethod
    def update_buildings(building, building_positions, worker_position):
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                if (i, j) not in building_positions:
                    building_positions[(i, j)] = building


def main(debug_env: bool, **kwargs):
    Env(rank=0, eval_steps=500, **kwargs).main()


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser()
    Env.add_arguments(PARSER)
    PARSER.add_argument("--random-seed", default=0, type=int)
    main(**vars(PARSER.parse_args()))
