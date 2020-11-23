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
    State,
    Action,
    BuildOrder,
    Coord,
    WorkerAction,
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
        return DebugBuildOrder(
            self.action1.building, location=(self.action1.i, self.action1.j)
        )

    def is_op(self):
        return True


class Env(env.Env):
    @staticmethod
    def building_allowed(
        building,
        building_positions,
        insufficient_resources,
        positions,
        assignment_location,
    ) -> bool:
        if assignment_location in building_positions:
            return False
        if building is Building.ASSIMILATOR:
            return assignment_location == positions[Resource.GAS]
        else:
            return assignment_location not in (
                *building_positions,
                positions[Resource.GAS],
                positions[Resource.MINERALS],
            )

    @staticmethod
    def compound_action(*args, **kwargs) -> DebugCompoundAction:
        return DebugCompoundAction(*args, **kwargs)

    def done_generator(self, *lines):
        while True:
            yield True, lambda: None

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
