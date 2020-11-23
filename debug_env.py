from dataclasses import dataclass

import data_types
import env
from data_types import ActionType, Building, X, WorkerID, Assignment, CompoundAction


@dataclass(frozen=True)
class DebugAction(data_types.Action):
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


@dataclass(frozen=True)
class DebugCompoundAction(data_types.CompoundAction):
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
        return self.action1.building


class Env(env.Env):
    @staticmethod
    def building_allowed(*args, **kwargs):
        return True

    @staticmethod
    def compound_action(*args, **kwargs) -> DebugCompoundAction:
        return DebugCompoundAction(*args, **kwargs)


def main(**kwargs):
    Env(rank=0, **kwargs).main()


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser()
    Env.add_arguments(PARSER)
    PARSER.add_argument("--seed", default=0, type=int)
    main(**vars(PARSER.parse_args()))
