from collections import Counter, namedtuple
import itertools
import re

import gym
from gym import spaces
from gym.envs.registration import EnvSpec
from gym.spaces import Box
from gym.utils import seeding
import numpy as np
import six

from ppo.utils import set_index
from rl_utils import cartesian_product

Subtask = namedtuple("Subtask", "interaction count object")
Obs = namedtuple("Obs", "base subtask subtasks next_subtask")


class SubtasksGridWorld(gym.Env):
    def __init__(
        self,
        text_map,
        min_objects,
        n_obstacles,
        random_obstacles,
        n_subtasks,
        interactions,
        max_task_count,
        object_types,
        evaluation=False,
        eval_subtasks=None,
        task=None,
    ):
        super().__init__()
        if eval_subtasks is None:
            eval_subtasks = []
        self.eval_subtasks = np.array(eval_subtasks)
        self.spec = EnvSpec
        self.n_subtasks = n_subtasks
        self.n_obstacles = n_obstacles
        self.min_objects = min_objects
        self.np_random = np.random
        self.transitions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])

        # self.state_char = '🚡'
        self.desc = np.array([list(r) for r in text_map])

        self.interactions = np.array(interactions)
        self.max_task_count = max_task_count
        self.object_types = np.array(object_types)
        self.random_task = task is None
        self.random_obstacles = random_obstacles

        # set on initialize
        self.initialized = False
        self.next_subtask = False
        self.obstacles_one_hot = np.zeros(self.desc.shape, dtype=bool)
        self.open_spaces = None
        self.obstacles = None

        self.possible_subtasks = np.array(
            list(
                itertools.product(
                    range(len(interactions)),
                    range(max_task_count),
                    range(len(object_types)),
                )
            )
        )
        possible_subtasks = np.expand_dims(self.possible_subtasks, 0)
        if eval_subtasks:
            in_eval = possible_subtasks == np.expand_dims(eval_subtasks, 1)
            in_eval = in_eval.all(axis=-1).any(axis=0)
            if evaluation:
                self.possible_subtasks = self.possible_subtasks[in_eval]
            else:
                not_in_eval = np.logical_not(in_eval)
                self.possible_subtasks = self.possible_subtasks[not_in_eval]

        def encode_task():
            for string in task:
                subtask = Subtask(*re.split("[\s\\\]+", string))
                yield (
                    list(self.interactions).index(subtask.interaction),
                    int(subtask.count),
                    list(self.object_types).index(subtask.object),
                )

        # set on reset:
        if task:
            self.subtasks = np.array(list(encode_task()))
        else:
            self.subtasks = None
        self.subtask_idx = 0
        self.count = None
        self.objects = None
        self.pos = None
        self.last_terminal = False
        self.last_action = None

        h, w = self.desc.shape

        self.observation_space = spaces.Dict(
            Obs(
                base=Box(
                    0,
                    1,
                    shape=(
                        1 + 1 + 1 + len(object_types),  # obstacles  # ice  # agent
                        h,
                        w,
                    ),
                ),
                subtask=spaces.Discrete(n_subtasks),
                subtasks=spaces.MultiDiscrete(
                    np.tile(
                        np.array(
                            [len(interactions), max_task_count, len(object_types)]
                        ),
                        (n_subtasks, 1),
                    )
                ),
                next_subtask=spaces.Discrete(2),
            )._asdict()
        )
        self.action_space = spaces.Discrete(len(self.transitions) + 2)
        world = self

        class _Subtask(Subtask):
            def __str__(self):
                string = f"{world.interactions[self.interaction]} {self.count + 1} {world.object_types[self.object]}"
                if self.count > 0:
                    string += "s"
                return string

        self.Subtask = _Subtask
        self.object_one_hots = np.vstack(
            [
                np.eye(1 + len(self.object_types)),
                np.zeros((1, 1 + len(self.object_types))),
            ]
        )
        self.layer_one_hots = np.eye(h * w).reshape(-1, h, w)

    @property
    def subtask(self):
        try:
            return self.subtasks[self.subtask_idx]
        except IndexError:
            return None

    def randomize_obstacles(self):
        h, w = self.desc.shape
        choices = cartesian_product(np.arange(h), np.arange(w))
        choices = choices[np.all(choices % 2 != 0, axis=-1)]
        randoms = self.np_random.choice(
            len(choices), replace=False, size=self.n_obstacles
        )
        self.obstacles = choices[randoms]
        self.obstacles_one_hot[:] = 0
        set_index(self.obstacles_one_hot, self.obstacles, True)
        self.obstacles = np.array(list(self.obstacles))

    def initialize(self):
        self.randomize_obstacles()
        h, w = self.desc.shape
        ij = cartesian_product(np.arange(h), np.arange(w))
        self.open_spaces = ij[
            np.logical_not(np.all(np.isin(ij, self.obstacles), axis=-1))
        ]
        self.initialized = True

    @property
    def transition_strings(self):
        return np.array(list("👆👇👈👉pt"))

    def render(self, mode="human", sleep_time=0.5):
        print("task:")
        self.render_task()
        if self.subtask is None:
            print("*************")
            print("Task Complete")
            print("*************")
        else:
            print("subtask:")
            self.render_current_subtask()
        if self.count is not None:
            print("remaining:", self.count + 1)
        print("action:", end=" ")
        if self.last_action is not None:
            print(self.transition_strings[self.last_action])
        else:
            print("reset")

        # noinspection PyTypeChecker
        desc = self.desc.copy()
        desc[self.obstacles_one_hot] = "#"
        for pos, obj in self.objects.items():
            desc[pos] = np.append(self.object_types, "i")[obj][0]
        desc[tuple(self.pos)] = "*"

        for row in desc:
            print(six.u(f"\x1b[47m\x1b[30m"), end="")
            print("".join(row), end="")
            print(six.u("\x1b[49m\x1b[39m"))
        # time.sleep(4 * sleep_time if self.last_terminal else sleep_time)

    def render_current_subtask(self):
        print(f"{self.subtask_idx}:{self.subtask}")

    def render_task(self):
        for line in self.subtasks:
            print(line)
        print()

    def subtasks_generator(self):
        last_subtask = None
        for _ in range(self.n_subtasks):
            possible_subtasks = self.possible_subtasks
            if last_subtask is not None:
                subset = np.any(self.possible_subtasks != last_subtask, axis=-1)
                possible_subtasks = possible_subtasks[subset]
            choice = self.np_random.choice(len(possible_subtasks))
            last_subtask = possible_subtasks[choice]
            yield self.Subtask(*last_subtask)

    def get_required_objects(self, task):
        for subtask in task:
            yield from [subtask.object] * (subtask.count + 1)

    def reset(self):
        if not self.initialized:
            self.initialize()
        elif self.random_obstacles:
            self.randomize_obstacles()

        if self.random_task:
            self.subtasks = list(self.subtasks_generator())
        types = list(self.get_required_objects(self.subtasks))
        n_random = max(len(types), self.min_objects)
        random_types = self.np_random.choice(
            len(self.object_types), replace=True, size=n_random - len(types)
        )
        types = np.concatenate([random_types, types])
        self.np_random.shuffle(types)

        randoms = self.np_random.choice(
            len(self.open_spaces), replace=False, size=n_random + 1  # + 1 for agent
        )
        *objects_pos, self.pos = self.open_spaces[randoms]

        self.objects = {tuple(p): t for p, t in zip(objects_pos, types)}

        self.subtask_idx = 0
        self.count = self.subtask.count
        self.last_terminal = False
        self.last_action = None
        return self.get_observation()

    def get_observation(self):
        agent_one_hot = np.zeros_like(self.desc, dtype=bool)
        set_index(agent_one_hot, self.pos, True)

        objects_desc = np.full(self.desc.shape, -1)
        for k, v in self.objects.items():
            objects_desc[k] = v

        obstacles = self.layer_one_hots[
            np.ravel_multi_index(self.obstacles.T, self.desc.shape)
        ].sum(0)
        objects = self.object_one_hots[objects_desc]
        agent = self.layer_one_hots[np.ravel_multi_index(self.pos, self.desc.shape)]

        obs = np.dstack(
            [
                np.expand_dims(obstacles, 2),
                objects,
                np.expand_dims(agent, 2),
                # np.dstack([obstacles, agent]),
            ]
        ).transpose(2, 0, 1)

        return Obs(
            base=obs,
            subtask=self.subtask_idx,
            subtasks=np.array([self.subtasks]),
            next_subtask=self.next_subtask,
        )._asdict()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, a):
        self.last_action = a
        self.next_subtask = False
        # act
        n_transitions = len(self.transitions)
        pos = tuple(self.pos)
        touching = pos in self.objects

        if touching:
            iterate = False
            object_type = self.objects[pos]
            interaction = self.interactions[self.subtask.interaction]
            if "visit" == interaction and object_type == self.subtask.object:
                iterate = True
            if a >= n_transitions:
                if a - n_transitions == 0:  # pick up
                    del self.objects[pos]
                    if "pick-up" == interaction and object_type == self.subtask.object:
                        iterate = True
                elif a - n_transitions == 1:  # transform
                    self.objects[pos] = len(self.object_types)
                    if (
                        "transform" == interaction
                        and object_type == self.subtask.object
                    ):
                        iterate = True
            if iterate:
                if self.count == 0:
                    self.subtask_idx = self.get_next_subtask()
                    self.next_subtask = True
                else:
                    self.count -= 1

        if a < n_transitions:
            # move
            pos = self.pos + self.transitions[a]
            if any(np.all(self.obstacles == pos, axis=-1)):
                pos = self.pos
            a_min = np.zeros(2)
            a_max = np.array(self.desc.shape) - 1
            self.pos = np.clip(pos, a_min, a_max).astype(int)

        self.last_terminal = t = self.subtask is None
        r = 1.0 if t else -0.1
        return self.get_observation(), r, t, {}

    def evaluate_condition(self):
        return self.conditions[self.subtask_idx] in self.objects.values()

    def get_next_subtask(self):
        return self.subtask_idx + 1


if __name__ == "__main__":
    import gym
    import gridworld_env.keyboard_control
    import gridworld_env.random_walk

    env = gym.make("4x4SubtasksGridWorld-v0")
    actions = "wsadeq"
    gridworld_env.keyboard_control.run(env, actions=actions)
