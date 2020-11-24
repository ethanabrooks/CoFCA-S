from dataclasses import dataclass

import env
import keyboard_control
from data_types import (
    ActionType,
    Building,
    X,
    WorkerID,
    Assignment,
    CompoundAction,
    Action,
    BuildOrder,
    Targets,
    WORLD_SIZE,
    Resource,
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
    i: X
    j: X

    @classmethod
    def num_values(cls) -> "DebugAction1":
        return cls(len(Building), i=WORLD_SIZE, j=WORLD_SIZE)

    def reset(self):
        return True

    def next_if_not_reset(self) -> ActionType:
        raise NotImplementedError

    @classmethod
    def parse(cls, a) -> "Action":
        parsed = super().parse(a)
        assert isinstance(parsed, DebugAction1)
        return cls(building=Building(1 + parsed.building), i=parsed.i, j=parsed.j)


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
        return BuildOrder(
            self.action1.building, location=(self.action1.i, self.action1.j)
        )

    def is_op(self):
        return True

    @staticmethod
    def initial_assignment():
        return Resource.GAS


class Env(env.Env):
    @staticmethod
    def building_allowed(
        building,
        building_positions,
        insufficient_resources,
        positions,
        assignment_location,
    ) -> bool:
        if insufficient_resources or assignment_location in building_positions:
            return False
        # if building is Building.ASSIMILATOR:
        #     return assignment_location == positions[Resource.GAS]
        # else:
        return assignment_location not in (
            *building_positions,
            positions[Resource.GAS],
            positions[Resource.MINERALS],
        )

    @staticmethod
    def compound_action(*args, **kwargs) -> DebugCompoundAction:
        return DebugCompoundAction(*args, **kwargs)

    def main(self):
        def action_fn(string: str):
            try:
                b, i, j = map(int, string.split())
                action1 = DebugAction1(building=Targets[b], i=i, j=j)
                act = self.compound_action(action1)
                return act
            except (ValueError, TypeError) as e:
                print(e)

        keyboard_control.run(self, action_fn)


def main(debug_env: bool, **kwargs):
    Env(rank=0, eval_steps=500, **kwargs).main()


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser()
    Env.add_arguments(PARSER)
    PARSER.add_argument("--random-seed", default=0, type=int)
    main(**vars(PARSER.parse_args()))
