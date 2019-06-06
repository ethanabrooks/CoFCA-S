import re
import time
from collections import namedtuple

import gym
import numpy as np
import six
from gym import spaces
from gym.utils import seeding

from ppo.utils import set_index
from rl_utils import cartesian_product

ObsSections = namedtuple('ObsSections', 'base subtask task next_subtask')


def get_task_space(task_types, max_task_count, object_types, n_subtasks):
    return spaces.MultiDiscrete(
        np.tile(
            np.array([len(task_types), max_task_count,
                      len(object_types)]), (n_subtasks, 1)))


class SubtasksGridWorld(gym.Env):
    def __init__(
            self,
            text_map,
            n_objects,
            n_obstacles,
            random_obstacles,
            n_subtasks,
            task_types,
            max_task_count,
            object_types,
            task=None,
    ):
        super().__init__()
        self.n_subtasks = n_subtasks
        self.n_obstacles = n_obstacles
        self.n_objects = n_objects
        self.np_random = np.random
        self.object_types = np.array(object_types)
        self.transitions = np.array([
            [-1, 0],
            [1, 0],
            [0, -1],
            [0, 1],
        ])

        # self.state_char = '🚡'
        self.desc = np.array([list(r) for r in text_map])

        self.task_types = np.array(task_types)

        self.max_task_count = max_task_count
        self.random_task = task is None
        self.random_obstacles = random_obstacles

        # set on initialize
        self.initialized = False
        self.obstacles_one_hot = np.zeros(self.desc.shape, dtype=bool)
        self.open_spaces = None
        self.obstacles = None

        def encode_task():
            for string in task:
                task_type, count, obj_type = re.split('[\s\\\]+', string)
                yield (list(self.task_types).index(task_type), int(count),
                       list(self.object_types).index(obj_type))

        # set on reset:
        if task:
            self.task = np.array(list(encode_task()))
        else:
            self.task = None
        self.subtask = None
        self.task_iter = None
        self.task_count = None
        self.objects = None
        self.pos = None
        self.last_terminal = False
        self.next_subtask = False

        h, w = self.desc.shape
        self.observation_space = spaces.Tuple([
            spaces.MultiDiscrete(
                np.ones((
                    1 +  # obstacles
                    1 +  # ice
                    1 +  # agent
                    len(object_types),
                    h,
                    w))),
            get_task_space(
                task_types=self.task_types,
                max_task_count=self.max_task_count,
                object_types=object_types,
                n_subtasks=n_subtasks)
        ])
        self.action_space = spaces.Discrete(len(self.transitions) + 2)

    def randomize_obstacles(self):
        h, w = self.desc.shape
        choices = cartesian_product(np.arange(h), np.arange(w))
        choices = choices[np.all(choices % 2 != 0, axis=-1)]
        randoms = self.np_random.choice(
            len(choices), replace=False, size=self.n_obstacles)
        self.obstacles = choices[randoms]
        self.obstacles_one_hot[:] = 0
        set_index(self.obstacles_one_hot, self.obstacles, True)
        self.obstacles = np.array(list(self.obstacles))

    def initialize(self):
        self.randomize_obstacles()
        h, w = self.desc.shape
        ij = cartesian_product(np.arange(h), np.arange(w))
        self.open_spaces = ij[np.logical_not(
            np.all(np.isin(ij, self.obstacles), axis=-1))]
        self.initialized = True

    @property
    def transition_strings(self):
        return np.array(list('🛑👇👆👉👈✋👊'))

    def render(self, mode='human'):
        def print_subtask(task_type, count, task_object_type):
            print(self.task_types[task_type], count,
                  self.object_types[task_object_type])

        print('task:')
        for task in self.task:
            print_subtask(*task)
        print()
        print('subtask:')
        print_subtask(*self.subtask)
        print('remaining:', self.task_count)

        # noinspection PyTypeChecker
        desc = self.desc.copy()
        desc[self.obstacles_one_hot] = '#'
        positions = self.objects_one_hot()
        types = np.append(self.object_types, 'ice')
        for pos, obj in zip(positions, types):
            desc[pos] = obj[0]
        desc[tuple(self.pos)] = '*'

        for row in desc:
            print(six.u(f'\x1b[47m\x1b[30m'), end='')
            print(''.join(row), end='')
            print(six.u('\x1b[49m\x1b[39m'))

        time.sleep(2 if self.last_terminal else .5)

    def reset(self):
        if not self.initialized:
            self.initialize()
        elif self.random_obstacles:
            self.randomize_obstacles()

        if self.random_task:
            task_types = self.np_random.choice(
                len(self.task_types), size=self.n_subtasks)
            task_objects = self.np_random.choice(
                len(self.object_types), size=self.n_subtasks)
            task_counts = self.np_random.choice(
                self.max_task_count, size=self.n_subtasks) + 1
            task_counts[self.task_types[task_types] == 'visit'] = 1
            self.task = np.stack([task_types, task_counts, task_objects],
                                 axis=1)
        self.task_iter = iter(self.task)

        types = [x for t, c, o in self.task for x in c * [o]]
        n_random = max(len(types), self.n_objects)
        random_types = self.np_random.choice(
            len(self.object_types), replace=True, size=n_random - len(types))
        types = np.concatenate([random_types, types])
        self.np_random.shuffle(types)

        randoms = self.np_random.choice(
            len(self.open_spaces),
            replace=False,
            size=n_random + 1  # + 1 for agent
        )
        *objects_pos, self.pos = self.open_spaces[randoms]

        self.objects = {tuple(p): t for p, t in zip(objects_pos, types)}

        self.task_count = None
        self.perform_iteration()
        return self.get_observation()

    def objects_one_hot(self):
        h, w, = self.desc.shape
        objects_one_hot = np.zeros((1 + len(self.object_types), h, w),
                                   dtype=bool)
        idx = [(v, ) + k for k, v in self.objects.items()]
        set_index(objects_one_hot, idx, True)
        return objects_one_hot

    def get_observation(self):
        agent_one_hot = np.zeros_like(self.desc, dtype=bool)
        set_index(agent_one_hot, self.pos, True)

        obs = [
            np.expand_dims(self.obstacles_one_hot, 0),
            self.objects_one_hot(),
            np.expand_dims(agent_one_hot, 0)
        ]

        # noinspection PyTypeChecker
        return np.vstack(obs), self.task

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def perform_iteration(self):
        self.next_subtask = self.task_count == 1
        if self.task_count is None or self.next_subtask:
            task_type, task_count, _ = self.subtask = next(self.task_iter)
            self.task_count = task_count
        else:
            self.task_count -= 1

    def step(self, a):
        self.next_subtask = False
        # act
        n_transitions = len(self.transitions)
        if a < n_transitions:
            # move
            pos = self.pos + self.transitions[a]
            if any(np.all(self.obstacles == pos, axis=-1)):
                pos = self.pos
            a_min = np.zeros(2)
            a_max = np.array(self.desc.shape) - 1
            self.pos = np.clip(pos, a_min, a_max).astype(int)
        pos = tuple(self.pos)
        touching = pos in self.objects

        t = False
        r = -.1
        if touching:
            iterate = False
            object_type = self.objects[pos]
            task_type_idx, _, task_object_type_idx = self.subtask
            task_type = self.task_types[task_type_idx]
            if 'visit' == task_type:
                iterate = object_type == task_object_type_idx
            if a >= n_transitions:
                if a - n_transitions == 0:  # pick up
                    del self.objects[pos]
                    if 'pick-up' == task_type:
                        iterate = object_type == task_object_type_idx  # picked up object
                elif a - n_transitions == 1:  # transform
                    self.objects[pos] = len(self.object_types)
                    if 'transform' == task_type:
                        iterate = object_type == task_object_type_idx

            if iterate:
                try:
                    self.perform_iteration()
                except StopIteration:
                    r = 1
                    t = True

        self.last_terminal = t
        return self.get_observation(), r, t, {}


if __name__ == '__main__':
    import gym
    import gridworld_env.keyboard_control
    import gridworld_env.random_walk
    from ppo.wrappers import SubtasksWrapper

    env = SubtasksWrapper(gym.make('4x4SubtasksGridWorld-v0'))
    actions = 'wsadeq'
    gridworld_env.keyboard_control.run(env, actions=actions)
