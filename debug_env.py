import typing
from collections import Counter
from dataclasses import astuple
from pprint import pprint
from typing import Union, Dict, Generator, Tuple, List

import numpy as np
from colored import fg

import env
from data_types import (
    Command,
    Resource,
    Building,
    Coord,
    Costs,
    Action,
    WorldObject,
    Movement,
    WorkerID,
    Worker,
    State,
    Line,
    BuildOrder,
)
from utils import RESET


class Env(env.Env):
    @staticmethod
    def building_allowed(building_positions, target_position, *args, **kwargs):
        return target_position not in building_positions

    def state_generator(self, *lines: Line) -> Generator[State, Action, None]:
        positions: List[Tuple[WorldObject, np.ndarray]] = [*self.place_objects()]
        building_positions: Dict[Coord, Building] = dict(
            [((i, j), b) for b, (i, j) in positions if isinstance(b, Building)]
        )
        positions: Dict[Union[Resource, WorkerID], Coord] = dict(
            [(o, (i, j)) for o, (i, j) in positions if not isinstance(o, Building)]
        )
        workers: Dict[WorkerID, Worker] = {}
        for worker_id in WorkerID:
            workers[worker_id] = Worker(assignment=Resource.MINERALS)

        required = Counter(l.building for l in lines if l.required)
        resources: typing.Counter[Resource] = Counter()
        ptr: int = 0
        while True:
            success = not required - Counter(building_positions.values())

            state = State(
                building_positions=building_positions,
                positions=positions,
                resources=resources,
                workers=workers,
                success=success,
                pointer=ptr,
            )

            def render():
                print("Resources:")
                pprint(resources)

            self.render_thunk = render

            nexus_positions: List[Coord] = [
                p for p, b in building_positions.items() if b is Building.NEXUS
            ]
            assert nexus_positions
            for worker_id, worker in workers.items():
                worker.next_action = worker.get_action(
                    position=positions[worker_id],
                    positions=positions,
                    nexus_positions=nexus_positions,
                )

            action: Action
            # noinspection PyTypeChecker
            action = yield state, render
            ptr += action.delta
            action = action.parse(tuple(self.world_shape))

            if isinstance(action, Command):
                workers[action.worker].assignment = action.assignment

            worker_id: WorkerID
            worker: Worker
            for worker_id, worker in sorted(
                workers.items(), key=lambda w: isinstance(w[1].assignment, BuildOrder)
            ):  # collect resources first.
                worker_position = positions[worker_id]
                if isinstance(worker.assignment, Resource):
                    resource = worker.assignment
                    resources[resource] += 1
                elif isinstance(worker.assignment, BuildOrder):
                    building = worker.assignment.building
                    if self.building_allowed(
                        building=building,
                        building_positions=building_positions,
                        insufficient_resources=False,
                        positions=positions,
                        target_position=worker.assignment.location,
                    ):
                        building_positions[worker.assignment.location] = building
                        # resources -= Costs[building]
                else:
                    raise RuntimeError


def main(debug_env, **kwargs):
    Env(rank=0, eval_steps=500, **kwargs).main()


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser()
    Env.add_arguments(PARSER)
    PARSER.add_argument("--random-seed", default=0, type=int)
    main(**vars(PARSER.parse_args()))
