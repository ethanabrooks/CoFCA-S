from dataclasses import dataclass
from typing import Generator

import env
import keyboard_control
from data_types import (
    ActionType,
    X,
    Worker,
    Assignment,
    CompoundAction,
    Action,
    Targets,
    WORLD_SIZE,
    Resource,
    Coord,
    Building,
)


@dataclass(frozen=True)
class DebugAction(Action):
    def next(self) -> ActionType:
        if self.reset():
            return DebugAction1
        return self.next_if_not_reset()


@dataclass(frozen=True)
class DebugAction1(DebugAction):
    target: X
    worker: X

    def to_ints(self) -> Generator[int, None, None]:
        yield Targets.index(self.target)
        yield self.worker.value - 1

    @classmethod
    def num_values(cls) -> "DebugAction1":
        return cls(target=len(Targets), worker=len(Worker))

    def reset(self):
        return isinstance(self.target, Resource)

    def next_if_not_reset(self) -> ActionType:
        return DebugAction2

    @classmethod
    def parse(cls, a) -> "Action":
        parsed = super().parse(a)
        assert isinstance(parsed, DebugAction1)
        return cls(target=Targets[parsed.target], worker=Worker(parsed.worker + 1))


@dataclass(frozen=True)
class DebugAction2(DebugAction):
    i: X
    j: X

    def to_ints(self) -> Generator[int, None, None]:
        yield self.i
        yield self.j

    @classmethod
    def num_values(cls) -> "DebugAction2":
        return cls(i=WORLD_SIZE, j=WORLD_SIZE)

    def reset(self):
        return True

    def next_if_not_reset(self) -> ActionType:
        raise NotImplementedError

    @classmethod
    def parse(cls, a) -> "Action":
        parsed = super().parse(a)
        assert isinstance(parsed, DebugAction2)
        return cls(i=parsed.i, j=parsed.j)


@dataclass(frozen=True)
class DebugCompoundAction(CompoundAction):
    action1: DebugAction1 = None
    action2: DebugAction2 = None
    ptr: int = 0
    active: ActionType = DebugAction1

    @classmethod
    def classes(cls):
        yield DebugAction1
        yield DebugAction2

    def actions(self):
        yield self.action1
        yield self.action2

    def coord(self) -> Coord:
        assert isinstance(self.action2, DebugAction2)
        return self.action2.i, self.action2.j

    def worker(self) -> Worker:
        assert isinstance(self.action1.worker, Worker)
        return self.action1.worker

    def assignment(self) -> Assignment:
        if isinstance(self.action1.target, Building):
            assert isinstance(self.action2, DebugAction2)
        return self.action1.target.assignment(
            None if self.action2 is None else self.coord()
        )


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
                ints = [*map(int, string.split())]
                try:
                    b, i, j = ints
                    action1 = DebugAction1(target=Targets[b], worker=Worker.A)
                    action2 = DebugAction2(i=i, j=j)
                except ValueError:
                    (r,) = ints
                    action1 = DebugAction1(target=Targets[r], worker=Worker.B)
                    action2 = None
                return self.compound_action(action1, action2)
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
