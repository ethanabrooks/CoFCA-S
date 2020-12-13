import copy
import typing
from collections import Counter, defaultdict
from dataclasses import astuple, dataclass
from typing import Union, Dict, Generator, Tuple, List, Optional
from pprint import pprint

import numpy as np

import env
from data_types import (
    ActionType,
    X,
    Action,
)
from data_types import (
    Resource,
    Building,
    Coord,
    WorldObject,
    Movement,
    Worker,
    State,
    Line,
    BuildOrder,
    CompoundAction,
    Assignment,
    Targets,
    WorkerAction,
    WORLD_SIZE,
    Nexus,
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
    ptr: int = 0
    active: ActionType = DebugAction1

    @classmethod
    def classes(cls):
        yield DebugAction1

    def actions(self):
        yield self.action1

    def worker(self) -> Worker:
        assert isinstance(self.action1.worker, Worker)
        return self.action1.worker

    def assignment(self) -> Assignment:
        if isinstance(self.action1.target, Building):
            assert isinstance(self.action2, DebugAction2)
        return self.action1.target.assignment(None)


@dataclass
class Env(env.Env):
    def state_generator(
        self, lines: List[Line], dependencies: Dict[Building, Building]
    ) -> Generator[State, CompoundAction, None]:
        positions: List[Tuple[WorldObject, np.ndarray]] = [*self.place_objects()]
        initial_buildings = dict(
            ((i, j), b) for b, (i, j) in positions if isinstance(b, Building)
        )
        building_positions: Dict[Coord, Building] = copy.deepcopy(initial_buildings)
        positions: Dict[Union[Resource, Worker], Coord] = dict(
            [(o, (i, j)) for o, (i, j) in positions if not isinstance(o, Building)]
        )
        assignments: Dict[Worker, Assignment] = {}
        next_actions: Dict[Worker, WorkerAction] = {}
        for worker_id in Worker:
            assignments[worker_id] = self.initial_assignment()

        required = Counter(li.building for li in lines if li.required)
        resources: typing.Counter[Resource] = Counter()
        ptr: int = 0
        action = self.compound_action()
        time_remaining = (
            self.eval_steps - 1
            if self.evaluating
            else (1 + len(lines)) * self.time_per_line
        )
        complete: typing.Counter[Building] = Counter()

        while True:
            buildings = Counter(self.get_buildings(building_positions)) - Counter(
                self.get_buildings(initial_buildings)
            )
            success = not required - buildings

            state = State(
                building_positions=building_positions,
                next_action=next_actions,
                positions=positions,
                resources=resources,
                success=success,
                pointer=ptr,
                action=action,
                time_remaining=time_remaining,
            )

            def render():
                print("Time remaining:", time_remaining)
                print("Complete:", complete)
                pprint(assignments)

            self.render_thunk = render

            for worker_id, assignment in assignments.items():
                next_actions[worker_id] = assignment.action(
                    positions[worker_id],
                    positions,
                    # [p for p, b in building_positions.items() if isinstance(b, Nexus)],
                    [positions[worker_id]],
                )

            action: CompoundAction
            # noinspection PyTypeChecker
            action = yield state, render
            assignment = action.assignment()
            dependency = (
                dependencies[assignment.building]
                if isinstance(assignment, BuildOrder)
                else None
            )
            if (
                isinstance(assignment, BuildOrder)
                and dependency is None
                or bool(complete[dependency])
            ):
                complete.update([assignment.building])
            ptr = action.ptr
            time_remaining -= 1
            assignments[action.worker()] = action.assignment()

            worker_id: Worker
            assignment: Assignment
            for worker_id, assignment in sorted(
                assignments.items(), key=lambda w: isinstance(w[1], BuildOrder)
            ):  # collect resources first.
                worker_position = positions[worker_id]
                worker_action = assignment.action(
                    current_position=worker_position,
                    positions=positions,
                    nexus_positions=[worker_position],
                )

                if isinstance(worker_action, Movement):
                    new_position = tuple(
                        np.array(worker_position) + np.array(astuple(worker_action))
                    )
                    positions[worker_id] = new_position
                    if isinstance(building_positions.get(new_position, None), Nexus):
                        for resource in Resource:
                            if self.gathered_resource(
                                building_positions, positions, resource, worker_position
                            ):
                                resources[resource] += 1
                elif isinstance(worker_action, Building):
                    building = worker_action
                    insufficient_resources = bool(
                        building.cost.as_counter() - resources
                    )
                    assert positions[worker_id] == assignment.location
                    allowed = self.building_allowed(
                        building=building,
                        dependency=dependencies[building],
                        building_positions=building_positions,
                        insufficient_resources=insufficient_resources,
                        positions=positions,
                        assignment_location=assignment.location,
                    )
                    if allowed:
                        building_positions[assignment.location] = building
                        resources -= building.cost.as_counter()
                else:
                    raise RuntimeError

    def building_allowed(
        self,
        building: Building,
        dependency: Optional[Building],
        building_positions: Dict[Coord, Building],
        insufficient_resources: bool,
        positions: Dict[WorldObject, Coord],
        assignment_location: Coord,
    ) -> bool:
        built = self.get_buildings(building_positions)
        # print(fg("green"), building, dependency, built, RESET)
        return dependency in built + [None] and assignment_location not in [
            *building_positions
        ]


def main(debug_env: bool, **kwargs):
    Env(rank=0, eval_steps=500, **kwargs).main()


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser()
    Env.add_arguments(PARSER)
    PARSER.add_argument("--random-seed", default=0, type=int)
    main(**vars(PARSER.parse_args()))
