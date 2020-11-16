from collections import Counter
from pprint import pprint
import numpy as np

import upper_env
from enums import Interaction, Refined, Other, Terrain
from lines import Subtask
from upper_env import Action


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
