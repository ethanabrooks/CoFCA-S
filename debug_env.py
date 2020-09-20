from typing import List, Generator, Tuple, Optional

from gym import spaces
import numpy as np
from utils import hierarchical_parse_args

import env
import keyboard_control
from env import ObjectMap, Coord, Line, State, Action, Obs


class Env(env.Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = spaces.Dict(
            Obs(
                active=spaces.Discrete(self.n_lines + 1),
                inventory=spaces.MultiBinary(len(self.items)),
                lines=spaces.MultiDiscrete(
                    np.array([len(self.possible_lines)] * self.n_lines)
                ),
                obs=spaces.Discrete(2),
            )._asdict()
        )

    def state_generator(
        self, objects: ObjectMap, agent_pos: Coord, lines: List[Line], **kwargs
    ) -> Generator[State, Tuple[int, int], None]:

        line_iterator = self.line_generator(lines)
        condition_bit = self.random.choice(2)
        subtask_iterator = self.subtask_generator(line_iterator, lines, condition_bit)
        prev, ptr = 0, next(subtask_iterator)
        term = False

        while True:
            state = State(
                obs=[condition_bit],
                prev=prev,
                ptr=ptr,
                term=term,
                subtask_complete=True,
                time_remaining=0,
                counts=None,
                inventory=None,
            )
            subtask_id, lower_level_index = yield state
            term = subtask_id != self.subtasks.index(lines[ptr].id)
            condition_bit = self.random.choice(2)
            prev, ptr = ptr, subtask_iterator.send(condition_bit)

    def evaluate_line(self, *args, condition_bit, **kwargs) -> bool:
        return bool(condition_bit)

    def populate_world(self, lines) -> Optional[Tuple[Coord, ObjectMap]]:
        return (0, 0), {}

    def feasible(self, objects, lines) -> bool:
        return True


def main(env: Env):
    def action_fn(string):
        try:
            action = int(string)
            if action > env.num_subtasks:
                raise ValueError
        except ValueError:
            return None

        return np.array(Action(upper=0, lower=action, delta=0, dg=0, ptr=0))

    keyboard_control.run(env, action_fn=action_fn)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    env.build_parser(parser)
    parser.add_argument("--seed", default=0, type=int)
    main(Env(rank=0, lower_level=None, **hierarchical_parse_args(parser)))
