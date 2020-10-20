from collections import Counter
from pprint import pprint

import upper_env
from enums import Interaction, Refined, Other, Terrain
from lines import Subtask
from upper_env import Action


class Env(upper_env.Env):
    def time_per_subtask(self):
        return 1

    def state_generator(self, *blocks):
        rooms = self.build_rooms(*blocks)
        assert len(rooms) == len(blocks)
        rooms_iter = iter(rooms)
        blocks_iter = iter(blocks)

        def next_required():
            block = next(blocks_iter, None)
            if block is not None:
                for subtask in block:
                    if subtask.interaction == Interaction.COLLECT:
                        yield subtask.resource
                    elif subtask.interaction == Interaction.REFINE:
                        yield Refined(subtask.resource.value)
                    else:
                        assert subtask.interaction == Interaction.BUILD

        required = Counter(next_required())
        objects = dict(next(rooms_iter))
        agent_pos = int(self.random.choice(self.h)), int(self.random.choice(self.w - 1))
        build_supplies = Counter()
        inventory = self.initialize_inventory()
        success = False
        action = None
        subtasks_completed = set()
        room_complete = False

        while True:
            self.make_feasible(objects)
            if room_complete:
                build_supplies -= required  # build bridge
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
            upper_action = self.subtasks[int(action.upper)]
            assert isinstance(upper_action, Subtask)
            if upper_action.interaction == Interaction.COLLECT:
                build_supplies[upper_action.resource] += 1
                if self.random.random() < self.map_discovery_prob:
                    inventory.add(Other.MAP)
            elif upper_action.interaction == Interaction.REFINE:
                build_supplies[Refined(upper_action.resource.value)] += 1
                if self.random.random() < self.map_discovery_prob:
                    inventory.add(Other.MAP)
                    room_complete = True
                    inventory.remove(Other.MAP)

            elif upper_action.interaction == Interaction.CROSS:
                if upper_action.resource == Other.MAP and Other.MAP in inventory:
                    room_complete = True
                elif upper_action.resource == Terrain.WATER:
                    if required + Counter() == build_supplies + Counter():
                        room_complete = True


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
