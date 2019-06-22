import itertools
import re

import gym
from gym import spaces
from gym.envs.registration import EnvSpec
from gym.utils import seeding
import numpy as np
import six

from ppo.utils import set_index
from rl_utils import cartesian_product


def get_task_space(interactions, max_task_count, object_types, n_subtasks):
    return spaces.MultiDiscrete(
        np.tile(
            np.array([len(interactions), max_task_count,
                      len(object_types)]), (n_subtasks, 1)))


class SubtasksGridWorld(gym.Env):
    def __init__(self,
                 text_map,
                 n_objects,
                 n_obstacles,
                 random_obstacles,
                 n_subtasks,
                 interactions,
                 max_task_count,
                 object_types,
                 evaluation,
                 eval_subtasks,
                 task=None):
        super().__init__()
        self.eval_subtasks = np.array(eval_subtasks)
        self.spec = EnvSpec
        self.n_subtasks = n_subtasks
        self.n_obstacles = n_obstacles
        self.n_objects = n_objects
        self.np_random = np.random
        self.transitions = np.array([
            [-1, 0],
            [1, 0],
            [0, -1],
            [0, 1],
        ])

        # self.state_char = '🚡'
        self.desc = np.array([list(r) for r in text_map])

        self.interactions = np.array(interactions)
        self.max_task_count = max_task_count
        self.object_types = np.array(object_types)
        self.random_task = task is None
        self.random_obstacles = random_obstacles

        # set on initialize
        self.initialized = False
        self.obstacles_one_hot = np.zeros(self.desc.shape, dtype=bool)
        self.open_spaces = None
        self.obstacles = None

        self.possible_subtasks = np.array(
            list(
                itertools.product(
                    range(len(interactions)),
                    range(1, 1 + max_task_count),
                    range(len(object_types)),
                )))
        possible_subtasks = np.expand_dims(self.possible_subtasks, 0)
        eval_subtasks = np.expand_dims(eval_subtasks, 1)
        if eval_subtasks:
            in_eval = possible_subtasks == eval_subtasks
            in_eval = in_eval.all(axis=-1).any(axis=0)
            if evaluation:
                self.possible_subtasks = self.possible_subtasks[in_eval]
            else:
                not_in_eval = np.logical_not(in_eval)
                self.possible_subtasks = self.possible_subtasks[not_in_eval]

        def encode_task():
            for string in task:
                interaction, count, obj_type = re.split('[\s\\\]+', string)
                yield (list(self.interactions).index(interaction), int(count),
                       list(self.object_types).index(obj_type))

        # set on reset:
        if task:
            self.task = np.array(list(encode_task()))
        else:
            self.task = None
        self.subtask_idx = None
        self.subtask = None
        self.task_iter = None
        self.task_count = None
        self.objects = None
        self.pos = None
        self.last_terminal = False
        self.last_action = None
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
                interactions=self.interactions,
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
        return np.array(list('👆👇👈👉pt'))

    def render(self, mode='human', sleep_time=.5):
        def print_subtask(interaction, count, task_object_type):
            print(self.interactions[interaction], count,
                  self.object_types[task_object_type])

        print('task:')
        for task in self.task:
            print_subtask(*task)
        print()
        print('subtask:')
        print_subtask(*self.subtask)
        print('remaining:', self.task_count)
        print('action:', end=' ')
        if self.last_action is not None:
            print(self.transition_strings[self.last_action])
        else:
            print('reset')

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
        # time.sleep(4 * sleep_time if self.last_terminal else sleep_time)

    def subtask_generator(self):
        last_subtask = None
        while True:
            possible_subtasks = self.possible_subtasks
            if last_subtask is not None:
                subset = np.any(
                    self.possible_subtasks != last_subtask, axis=-1)
                possible_subtasks = possible_subtasks[subset]
            choice = self.np_random.choice(len(possible_subtasks))
            last_subtask = possible_subtasks[choice]
            yield last_subtask

    def reset(self):
        if not self.initialized:
            self.initialize()
        elif self.random_obstacles:
            self.randomize_obstacles()

        if self.random_task:
            task_iter = itertools.islice(self.subtask_generator(),
                                         self.n_subtasks)
            self.task = np.array(list(task_iter))
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
        self.subtask_idx = -1
        self.perform_iteration()
        self.last_terminal = False
        self.last_action = None
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
            self.subtask_idx += 1
            interaction, task_count, _ = self.subtask = next(self.task_iter)
            self.task_count = task_count
        else:
            self.task_count -= 1

    def step(self, a):
        self.last_action = a
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

        obs = self.get_observation()

        t = False
        r = -.1
        if touching:
            iterate = False
            object_type = self.objects[pos]
            interaction_idx, _, task_object_type_idx = self.subtask
            interaction = self.interactions[interaction_idx]
            if 'visit' == interaction:
                iterate = object_type == task_object_type_idx
            if a >= n_transitions:
                if a - n_transitions == 0:  # pick up
                    del self.objects[pos]
                    if 'pick-up' == interaction:
                        iterate = object_type == task_object_type_idx  # picked up object
                elif a - n_transitions == 1:  # transform
                    self.objects[pos] = len(self.object_types)
                    if 'transform' == interaction:
                        iterate = object_type == task_object_type_idx

            if iterate:
                try:
                    self.perform_iteration()
                except StopIteration:
                    r = 1
                    t = True

        self.last_terminal = t
        return obs, r, t, {}


if __name__ == '__main__':
    import gym
    import gridworld_env.keyboard_control
    import gridworld_env.random_walk
    from ppo.wrappers import SubtasksWrapper

    env = SubtasksWrapper(gym.make('4x4SubtasksGridWorld-v0'))
    actions = 'wsadeq'
    gridworld_env.keyboard_control.run(env, actions=actions)
