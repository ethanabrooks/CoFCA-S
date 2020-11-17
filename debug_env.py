from collections import Counter
from pprint import pprint
import numpy as np

from lines import If, While
from utils import hierarchical_parse_args, RESET

import env
import keyboard_control
from env import ObjectMap, Coord, Line, State, Action, Obs


class Env(upper_env.Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space.spaces.update(
            inventory=spaces.MultiBinary(1),
            obs=spaces.Box(low=0, high=1, shape=(1, 1, 1), dtype=np.float32),
        )

    def srti_generator(
        self, objects: ObjectMap, agent_pos: Coord, lines: List[Line], **kwargs
    ) -> Generator[State, Tuple[int, int], None]:

        line_iterator = self.line_generator(lines)
        condition_bit = self.random.choice(2)
        subtask_iterator = self.subtask_generator(
            line_iterator, lines, condition_bit=condition_bit
        )
        prev, ptr = 0, next(subtask_iterator)
        term = False

        while True:
            self.make_feasible(objects)
            if room_complete:
                room = next(rooms_iter, None)
                if room is None:
                    success = True
                else:
                    objects = dict(room)
                required = Counter(next_required())

            state = dict(
                inventory=inventory,
                agent_pos=agent_pos,
                objects=objects,
                action=action,
                success=success,
                subtasks_completed=subtasks_completed,
                room_complete=room_complete,
                required=required,
            )
            subtask_id, lower_level_index = yield state
            term = subtask_id != self.subtasks.index(lines[ptr].id)
            condition_bit = self.random.choice(2)
            prev, ptr = ptr, subtask_iterator.send(dict(condition_bit=condition_bit))

    def inventory_representation(self, state):
        return np.array([0])

    def evaluate_line(self, line, loops, condition_bit, **kwargs) -> bool:
        return bool(condition_bit)

    def populate_world(self, lines) -> Optional[Tuple[Coord, ObjectMap]]:
        return (0, 0), {}

    def feasible(self, objects, lines) -> bool:
        return True

    def render_world(
        self,
        state,
        action,
        reward,
    ):
        if action is not None and action < len(self.subtasks):
            print("Selected:", self.subtasks[action], action)
        print("Action:", action)
        print("Reward", reward)
        for i, subtask in enumerate(self.subtasks):
            print(i, subtask)

    def render_instruction(
        self,
        term,
        success,
        lines,
        state,
        agent_ptr,
    ):

        if term:
            print(env.GREEN if success else env.RED)
        indent = 0
        for i, line in enumerate(lines):
            if i == state.ptr and i == agent_ptr:
                pre = "+ "
            elif i == agent_ptr:
                pre = "- "
            elif i == state.ptr:
                pre = "| "
            else:
                pre = "  "
            indent += line.depth_change[0]
            if type(line) in (If, While):
                evaluation = state.counts
                line_str = f"{line} {evaluation}"
            else:
                line_str = str(line)
            print("{:2}{}{}{}".format(i, pre, " " * indent, line_str))
            indent += line.depth_change[1]
        print("Condition bit:", state.counts)
        print(RESET)


def main(env: Env):
    def action_fn(string):
        try:
            action = int(string)
            if action > env.num_subtasks:
                raise ValueError
        except ValueError:
            return None

        return np.array(Action(upper=action, lower=0, delta=0, dg=0, ptr=0))

            def render():
                print("Inventory:")
                pprint(inventory)
                print("Build Supplies:")
                pprint(build_supplies)
                print("Required:")
                pprint(required)

            self.render_thunk = render
            # for name, space in self.observation_space.spaces.items():
            #     if not space.contains(s[name]):
            #         import ipdb
            #
            #         ipdb.set_trace()
            #         space.contains(s[name])
            action = yield state, render  # type: Action
            subtasks_completed = set()
            room_complete = False
            if self.random.random() < self.bandit_prob:
                possessions = [k for k, v in build_supplies.items() if v > 0]
                if possessions:
                    robbed = self.random.choice(possessions)
                    build_supplies[robbed] -= 1
            assert isinstance(action.upper, Subtask)
            if action.upper.interaction == Interaction.COLLECT:
                build_supplies[action.upper.resource] += 1
                subtasks_completed.add(action.upper)
                if self.random.random() < self.map_discovery_prob:
                    inventory.add(Other.MAP)
            elif action.upper.interaction == Interaction.REFINE:
                build_supplies[Refined(action.upper.resource.value)] += 1
                subtasks_completed.add(action.upper)
                if self.random.random() < self.map_discovery_prob:
                    inventory.add(Other.MAP)

            elif action.upper.interaction == Interaction.CROSS:
                if action.upper.resource == Terrain.MOUNTAIN and Other.MAP in inventory:
                    room_complete = True
                    inventory.remove(Other.MAP)
                    subtasks_completed.add(action.upper)
                elif action.upper.resource == Terrain.WATER:
                    if (
                        (
                            required + Counter() == build_supplies + Counter()
                        )  # inventory == required
                        if self.exact_count
                        else (
                            not required - build_supplies
                        )  # inventory dominates required
                    ):
                        build_supplies -= required  # build bridge
                        if self.random.random() > self.bridge_failure_prob:
                            room_complete = True
                            subtasks_completed.add(action.upper)


def main(lower_level_load_path, lower_level_config, debug_env, **kwargs):
    Env(rank=0, min_eval_lines=0, max_eval_lines=10, **kwargs).main(
        lower_level_load_path=lower_level_load_path,
        lower_level_config=lower_level_config,
    )


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser()
    Env.add_arguments(PARSER)
    PARSER.add_argument("--seed", default=0, type=int)
    PARSER.add_argument("--lower-level-config", default="lower.json")
    PARSER.add_argument("--lower-level-load-path", default="lower.pt")
    main(**vars(PARSER.parse_args()))
