import gym
import numpy as np
import six
from gym import spaces
from gym.utils import seeding
from gym.utils.colorize import color2num


def set_index(array, idxs, value):
    idxs = np.array(idxs)
    if idxs.size > 0:
        array[tuple(idxs.T)] = value


def get_index(array, idxs):
    idxs = np.array(idxs)
    if idxs.size == 0:
        return np.array([], array.dtype)
    return array[tuple(idxs.T)]


class LogicGridWorld(gym.Env):
    def __init__(self, objects, text_map, partial_obs=False):
        super().__init__()

        self.partial_obs = partial_obs
        self.np_random = np.random

        self.transitions = np.array([
            [-1, 0],
            [1, 0],
            [0, -1],
            [0, 1],
        ])

        # self.state_char = '🚡'
        self.state_char = '*'
        self.background = np.array([list(r) for r in text_map])
        self.objects_list = list(map(np.array, objects))
        self.objects = np.concatenate(objects)
        self.objects.sort()

        self.task_types = ['move', 'touch']
        self.colors = np.unique(self.background)
        self.one_hot_background = np.expand_dims(self.background, 2) == \
                                  self.colors.reshape((1, 1, -1))
        self.one_hot_background = self.one_hot_background.astype(bool)

        # set on reset:
        self.objects_pos = None
        self.object_grasped = None
        self.touched = None
        self.object_type = None
        self.task_color = None
        self.task_objects = None
        self.task_type_idx = None
        self.task_type = None
        self.target_color = None
        self.pos = None

        self.randomize_positions()
        self.observation_space = spaces.Box(low=0, high=1, shape=self.get_observation().shape)
        self.action_space = spaces.Discrete(len(self.transitions) + 1)

    @property
    def transition_strings(self):
        return np.array(list('🛑👇👆👉👈✋👊'))

    def render(self, mode='human'):
        print('touched:', list(self.objects[self.touched]))
        if self.task_type == 'touch':
            print('touch:', list(self.to_touch()))
        elif self.task_type == 'move':
            print('move:', self.to_move())
            print('to:', self.target_color)
        else:
            raise RuntimeError

        colors = dict(r='red', g='green', b='blue', y='yellow')

        desc = np.full_like(self.background, ' ')

        set_index(desc, self.pos, self.state_char)
        set_index(desc, self.objects_pos, self.objects)
        touching = self.objects[self.touching()]
        for back_row, front_row in zip(self.background, desc):
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

    def touching(self):
        return np.all(self.objects_pos == self.pos, axis=1)

    def to_move(self):
        object_colors = self.get_colors_for(self.task_objects)
        return self.task_objects[object_colors != self.target_color]

    def to_touch(self):
        task_touched = np.isin(self.task_objects, self.objects[self.touched])
        task_untouched = np.logical_not(task_touched)
        return self.task_objects[task_untouched]

    def get_observation(self):
        h, w, = self.background.shape
        objects_one_hot = np.zeros((h, w, self.objects.size), dtype=bool)
        idx = np.hstack([self.objects_pos,
                         np.expand_dims(np.arange(self.objects.size), 1)])
        set_index(objects_one_hot, idx, True)

        grasped_one_hot = np.zeros_like(self.background, dtype=bool)
        grasped_pos = self.objects_pos[self.object_grasped]
        set_index(grasped_one_hot, grasped_pos, True)

        agent_one_hot = np.zeros_like(self.background, dtype=bool)
        set_index(agent_one_hot, self.pos, True)

        obs = [self.one_hot_background, objects_one_hot, grasped_one_hot, agent_one_hot]
        if not self.partial_obs:
            dest_one_hot = np.zeros((h, w, self.colors.size + 1))
            todo_one_hot = np.zeros_like(self.background)
            if self.task_type == 'touch':
                dest_one_hot[:, :, -1] = True
                todo_objects = self.to_touch()
            elif self.task_type == 'move':
                dest_one_hot[:, :, :] = self.target_color == self.colors
                todo_objects = self.to_move()
            else:
                raise RuntimeError

            todo = np.isin(self.objects, todo_objects)
            todo_pos = self.objects_pos[todo]
            if todo_pos.size > 0:
                set_index(todo_one_hot, todo_pos, True)
            obs += [dest_one_hot, todo_one_hot]

        return np.dstack(obs).astype(float).flatten() # TODO

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_colors_for(self, objects):
        idxs = np.isin(self.objects, objects)
        if idxs.size == 0:
            return np.array([], self.background.dtype)
        return get_index(self.background, self.objects_pos[idxs])

    def step(self, a):
        n_transitions = len(self.transitions)
        if a < n_transitions:
            # move
            self.pos += self.transitions[a]
            a_min = np.zeros(2)
            a_max = np.array(self.background.shape) - 1
            self.pos = np.clip(self.pos, a_min, a_max).astype(int)
            if np.any(self.object_grasped):
                self.objects_pos[self.object_grasped] = self.pos
            touching = self.touching()
        else:
            # interact
            touching = self.touching()
            if any(touching):
                idx = touching.argmax()
                if self.object_grasped[idx]:
                    self.object_grasped[idx] = 0
                elif not any(self.object_grasped):
                    self.object_grasped[idx] = 1

        self.touched[touching] = True

        # reward / terminal
        if self.task_type == 'touch':
            success = np.all(np.isin(self.task_objects,
                                     self.objects[self.touched]))
        elif self.task_type == 'move':
            object_colors = self.get_colors_for(self.task_objects)
            success = np.all(object_colors == self.target_color)
        else:
            raise RuntimeError
        t = bool(success)
        r = float(success)

        return self.get_observation(), r, t, {}

    def randomize_positions(self):
        randoms = self.np_random.choice(self.background.size,
                                        replace=False,
                                        size=self.objects.size + 1, )
        self.pos, *self.objects_pos = zip(*np.unravel_index(randoms, self.background.shape))
        self.objects_pos = np.array(self.objects_pos)

    def reset(self):
        self.object_grasped = np.zeros_like(self.objects, dtype=bool)
        self.touched = np.zeros_like(self.objects, dtype=bool)
        self.randomize_positions()

        # task type
        self.task_type = self.np_random.choice(self.task_types)

        # task objects
        self.object_type = self.np_random.choice(len(self.objects_list))
        objects = self.objects_list[self.object_type]
        object_colors = self.get_colors_for(objects)
        self.task_color = self.np_random.choice(np.unique(object_colors))  # exclude empty colors
        self.task_objects = objects[object_colors == self.task_color]
        self.task_objects.sort()

        target_choices = self.colors
        task_obj_colors = self.get_colors_for(self.task_objects)
        if self.task_type == 'move' and task_obj_colors.size == 1:
            target_choices = target_choices[target_choices != task_obj_colors.item()]
        self.target_color = self.np_random.choice(target_choices)

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
