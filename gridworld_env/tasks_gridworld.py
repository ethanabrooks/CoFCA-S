import time

import gym
from gym import spaces
from gym.utils import seeding
from gym.utils.colorize import color2num
import numpy as np
import six


def set_index(array, idxs, value):
    idxs = np.array(idxs)
    if idxs.size > 0:
        array[tuple(idxs.T)] = value


def get_index(array, idxs):
    idxs = np.array(idxs)
    if idxs.size == 0:
        return np.array([], array.dtype)
    return array[tuple(idxs.T)]


class TasksGridWorld(gym.Env):
    def __init__(self, objects, text_map, partial=False):
        super().__init__()
        self.partial = partial
        self.np_random = np.random
        self.transitions = np.array([
            [-1, 0],
            [1, 0],
            [0, -1],
            [0, 1],
        ])

        # self.state_char = '🚡'
        self.state_char = '*'
        self.desc = np.array([list(r) for r in text_map])
        self.objects_list = list(map(np.array, objects))
        self.objects = np.concatenate(objects)

        self.task_types = ['visit',
                           'pick-up',
                           'transform',
                           'pick-up-2',
                           'transform-2',
                           'pick-up-3',
                           'transform-3', ]

        # set on reset:
        self.task_objects = None
        self.task_type = None
        self.pos = None
        self.task_object_type = None
        self.task_count = None
        self.tasks = None
        self.task = None

        self.reset()
        self.observation_space = spaces.Box(
            low=0, high=1, shape=self.get_observation().shape)
        self.action_space = spaces.Discrete(len(self.transitions) + 2)

    @property
    def transition_strings(self):
        return np.array(list('🛑👇👆👉👈✋👊'))

    def render(self, mode='human'):
        print('touched:', list(self.objects[self.touched]))

        colors = dict(r='red', g='green', b='blue', y='yellow')

        # noinspection PyTypeChecker
        desc = np.full_like(self.desc, ' ')

        set_index(desc, self.pos, self.state_char)
        set_index(desc, self.objects, self.objects)  # TODO
        touching = self.objects[self.touching()]
        for back_row, front_row in zip(self.desc, desc):
            print(six.u('\x1b[30m'), end='')
            last = None
            for back, front in zip(back_row, front_row):
                if back != last:
                    color = colors[back]
                    num = color2num[color] + 10
                    highlight = six.u(str(num))
                    print(six.u(f'\x1b[{highlight}m'), end='')

                if front in touching:
                    print(six.u('\x1b[7m'), end='')
                print(front, end='')
                if front in touching:
                    print(six.u('\x1b[27m'), end='')
                last = back
            print(six.u('\x1b[0m'))
        print(six.u('\x1b[39m'), end='')
        time.sleep(2 if self.last_terminal else .5)

    def get_observation(self):
        h, w, = self.desc.shape
        objects_one_hot = np.zeros((h, w, self.objects.size), dtype=bool)
        idx = np.hstack([
            self.objects,
            np.expand_dims(np.arange(self.objects.size), 1)
        ])
        set_index(objects_one_hot, idx, True)

        grasped_one_hot = np.zeros_like(self.desc, dtype=bool)
        grasped_pos = self.objects[self.object_grasped]
        set_index(grasped_one_hot, grasped_pos, True)

        agent_one_hot = np.zeros_like(self.desc, dtype=bool)
        set_index(agent_one_hot, self.pos, True)

        # partial observability stuff
        task_type_one_hot = np.zeros((h, w, len(self.task_types)), dtype=bool)
        task_type_one_hot[:, :, self.task_type_idx] = True

        task_objects_one_hot = np.zeros((h, w), dtype=bool)
        set_index(task_objects_one_hot, self.objects[self.task_object_type], True)

        obs = [
            objects_one_hot, grasped_one_hot,
            agent_one_hot, task_type_one_hot, task_objects_one_hot
        ]

        # noinspection PyTypeChecker
        return np.dstack(obs).astype(float).transpose(2, 0, 1)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, a):
        n_transitions = len(self.transitions)
        if a < n_transitions:
            # move
            self.pos += self.transitions[a]
            a_min = np.zeros(2)
            a_max = np.array(self.desc.shape) - 1
            self.pos = np.clip(self.pos, a_min, a_max).astype(int)
        touching = self.pos in self.objects
        object_type = self.objects[self.pos]
        picked_up = None
        transformed = None
        if a >= n_transitions and touching:
            if a - n_transitions == 0:  # pick up
                picked_up = self.objects[self.pos]
                del self.objects[self.pos]
            elif a - n_transitions == 1:  # transform
                transformed = self.objects[self.pos]
                object_type = 'ice'

        # reward / terminal
        if 'visit' == self.task_type:
            iterate = object_type == self.task_object_type
        elif 'pick-up' in self.task_type:
            iterate = picked_up == self.task_object_type
        elif 'transform' in self.task_type:
            iterate = transformed == self.task_object_type
        else:
            raise RuntimeError

        success = False
        if iterate:
            self.task_count -= 1
            if self.task_count == 0:
                try:
                    self.task = next(self.tasks)
                except StopIteration:
                    success = True

        t = bool(success)
        r = float(success)
        return self.get_observation(), r, t, {}

    def randomize_positions(self, objects_type):
        randoms = self.np_random.choice(
            self.desc.size,
            replace=False,
            size=len(self.objects) + 1,  # + 1 for agent
        )
        self.pos, *objects_pos = zip(
            *np.unravel_index(randoms, self.desc.shape))

        self.objects = dict(zip(objects_pos, objects_type))

    def reset(self):
        objects_type =
        self.randomize_positions()

        # task type
        self.task_type = self.np_random.choice(self.task_types)

        # task objects
        self.object_type = self.np_random.choice(len(self.objects_list))
        objects = self.objects_list[self.object_type]
        object_colors = self.get_colors_for(objects)
        self.task_color = self.np_random.choice(
            np.unique(object_colors))  # exclude empty colors
        self.task_objects = objects[object_colors == self.task_color]

        target_choices = self.colors
        task_obj_colors = np.unique(self.get_colors_for(self.task_objects))
        if self.task_type == 'move' and task_obj_colors.size == 1:
            target_choices = target_choices[
                target_choices != task_obj_colors.item()]
        self.target_color = self.np_random.choice(target_choices)

        self.last_to_touch = self.to_touch()
        self.last_to_move = self.to_move()
        self.last_terminal = False

        self._todo_one_hot = self.todo_one_hot()
        return self.get_observation()


if __name__ == '__main__':
    import gym
    import gridworld_env.keyboard_control
    import gridworld_env.random_walk

    env = gym.make('4x4FourSquareGridWorld-v0')
    # env = gym.make('1x4TwoSquareGridWorld-v0')
    actions = 'wsadx'
    gridworld_env.keyboard_control.run(env, actions=actions)
    # gridworld_env.random_walk.run(env)
