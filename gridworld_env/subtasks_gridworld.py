import time

import gym
import numpy as np
import six
from gym import spaces
from gym.utils import seeding

from rl_utils import cartesian_product


def set_index(array, idxs, value):
    idxs = np.array(idxs)
    if idxs.size > 0:
        array[tuple(idxs.T)] = value


def get_index(array, idxs):
    idxs = np.array(idxs)
    if idxs.size == 0:
        return np.array([], array.dtype)
    return array[tuple(idxs.T)]


class SubtasksGridWorld(gym.Env):
    def __init__(self,
                 object_types,
                 text_map,
                 n_objects,
                 n_obstacles,
                 n_subtasks,
                 partial=False):
        super().__init__()
        self.n_subtasks = n_subtasks
        self.n_obstacles = n_obstacles
        self.n_objects = n_objects
        self.partial = partial
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

        self.task_types = [
            'visit',
            'pick-up',
            'transform',
            'pick-up-2',
            'transform-2',
            'pick-up-3',
            'transform-3',
        ]

        self.task_counts = [
            1,
            1,
            1,
            2,
            2,
            3,
            3,
        ]

        self.initialized = False
        # set on initialize
        self.obstacles_one_hot = np.zeros(self.desc.shape, dtype=bool)
        self.open_spaces = None
        self.obstacles = None

        # set on reset:
        self.objects = None
        self.pos = None
        self.task_count = None
        self.tasks = None
        self.task = None
        self.iterate = None
        self.last_terminal = False

        self.reset()
        self.observation_space = spaces.Box(
            low=0, high=1, shape=self.get_observation().shape)
        self.action_space = spaces.Discrete(len(self.transitions) + 2)

    def initialize(self):
        h, w = self.desc.shape
        choices = cartesian_product(np.arange(h), np.arange(w))
        choices = choices[np.all(choices % 2 != 0, axis=-1)]
        randoms = self.np_random.choice(
            len(choices), replace=False, size=self.n_obstacles)
        self.obstacles = choices[randoms]
        set_index(self.obstacles_one_hot, self.obstacles, True)
        self.obstacles = np.array(list(self.obstacles))
        h, w = self.desc.shape
        ij = cartesian_product(np.arange(h), np.arange(w))
        self.open_spaces = ij[np.logical_not(
            np.all(np.isin(ij, self.obstacles), axis=-1))]
        self.initialized = True

    @property
    def transition_strings(self):
        return np.array(list('🛑👇👆👉👈✋👊'))

    def render(self, mode='human'):
        task_type, task_object_type = self.task
        print('task:', self.task_types[task_type],
              self.object_types[task_object_type])
        print('task count:', self.task_count + 1)

        # noinspection PyTypeChecker
        desc = self.desc.copy()
        desc[self.obstacles_one_hot] = '#'
        positions = self.objects_one_hot().transpose(2, 0, 1)
        types = np.append(self.object_types, 'ice')
        for pos, obj in zip(positions, types):
            desc[pos] = obj[0]
        desc[tuple(self.pos)] = '*'

        for row in desc:
            print(six.u(f'\x1b[47m'), end='')
            print(''.join(row), end='')
            print(six.u('\x1b[49m'))

        time.sleep(2 if self.last_terminal else .5)

    def reset(self):
        if not self.initialized:
            self.initialize()

        # tasks
        task_types = self.np_random.choice(
            len(self.task_types), size=self.n_subtasks)
        task_objects = self.np_random.choice(
            len(self.object_types), size=self.n_subtasks)
        tasks = np.stack([task_types, task_objects], axis=1)
        self.tasks = iter(tasks)

        types = [x for t, o in tasks for x in self.task_counts[t] * [o]]
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
        objects_one_hot = np.zeros((h, w, 1 + len(self.object_types)),
                                   dtype=bool)
        idx = [k + (v, ) for k, v in self.objects.items()]
        set_index(objects_one_hot, idx, True)
        return objects_one_hot

    def get_observation(self):
        h, w, = self.desc.shape

        agent_one_hot = np.zeros_like(self.desc, dtype=bool)
        set_index(agent_one_hot, self.pos, True)

        # partial observability stuff
        task_type, task_object_type = self.task
        task_type_one_hot = np.zeros((h, w, len(self.task_types)), dtype=bool)
        task_type_one_hot[:, :, task_type] = True

        # TODO: make this less easy
        task_objects_one_hot = np.zeros((h, w), dtype=bool)
        idx = [k for k, v in self.objects.items() if v == task_object_type]
        set_index(task_objects_one_hot, idx, True)

        obs = [
            self.obstacles_one_hot,
            self.objects_one_hot(), agent_one_hot, task_type_one_hot,
            task_objects_one_hot
        ]

        # noinspection PyTypeChecker
        transpose = np.dstack(obs).astype(float).transpose(2, 0, 1)

        # names = ['obstacles'] + list(self.object_types) + ['ice', 'agent'] + \
        #         list(self.task_types) + ['task objects']
        # assert len(transpose) == len(names)
        # for array, name in zip(transpose, names):
        #     print(name)
        #     print(array)

        return transpose

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def perform_iteration(self):
        self.iterate = False
        if not self.task_count:
            task_type, _ = self.task = next(self.tasks)
            self.task_count = self.task_counts[task_type]
        self.task_count -= 1

    def step(self, a):
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

        if touching:
            object_type = self.objects[pos]
            picked_up = transformed = None
            if a >= n_transitions:
                if a - n_transitions == 0:  # pick up
                    picked_up = object_type
                    del self.objects[pos]
                elif a - n_transitions == 1:  # transform
                    transformed = object_type
                    self.objects[pos] = len(self.object_types)

            # iterate
            task_type_idx, task_object_type_idx = self.task
            task_type = self.task_types[task_type_idx]
            if 'visit' == task_type:
                self.iterate = object_type == task_object_type_idx
            elif 'pick-up' in task_type:
                self.iterate = picked_up == task_object_type_idx
            elif 'transform' in task_type:
                self.iterate = transformed == task_object_type_idx
            else:
                raise RuntimeError

        # next task / terminate
        t = False
        if self.iterate:
            try:
                self.perform_iteration()
            except StopIteration:
                t = True

        self.last_terminal = t
        return self.get_observation(), -.1, t, {}


if __name__ == '__main__':
    import gym
    import gridworld_env.keyboard_control
    import gridworld_env.random_walk

    env = gym.make('4x4TasksGridWorld-v0')
    actions = 'wsadeq'
    gridworld_env.keyboard_control.run(env, actions=actions)
