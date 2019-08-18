from collections import namedtuple

import gym
import numpy as np
from gym import spaces
from gym.spaces import Box

from ppo.oh_et_al.gridworld import Obs

SubtasksActions = namedtuple("SubtasksActions", "a cr cg g")


class DebugWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_guess = None
        self.last_reward = None
        self.subtask_space = env.task_space.nvec[0]

    def step(self, action):
        action_sections = get_subtasks_action_sections(self.action_space.spaces)
        actions = SubtasksActions(
            *[int(x.item()) for x in np.split(action, np.cumsum(action_sections)[:-1])]
        )
        s, _, t, i = super().step(action)
        guess = int(actions.g)
        truth = int(self.env.unwrapped.subtask_idx)
        r = float(np.all(guess == truth)) - 1
        self.last_guess = guess
        self.last_reward = r
        return s, r, t, i

    def render(self, mode="human"):
        print("########################################")
        super().render(sleep_time=0)
        print("guess", self.last_guess)
        print("truth", self.env.unwrapped.subtask_idx)
        print("reward", self.last_reward)
        # input('pause')


class Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_space, task_space = env.observation_space.spaces
        assert np.all(task_space.nvec == task_space.nvec[0])
        _, h, w = obs_space.shape
        d = sum(get_subtasks_obs_sections(task_space))
        self.task_space = task_space
        self.observation_space = Box(0, 1, shape=(d, h, w))
        self.action_space = spaces.Tuple(
            SubtasksActions(
                a=env.action_space,
                g=spaces.Discrete(env.n_subtasks),
                cg=spaces.Discrete(2),
                cr=spaces.Discrete(2),
            )
        )
        self.last_g = None

    def step(self, action):
        action_sections = np.cumsum(
            get_subtasks_action_sections(self.action_space.spaces)
        )[:-1].astype(int)
        actions = SubtasksActions(*np.split(action, action_sections))
        action = int(actions.a)
        self.last_g = int(actions.g)
        s, r, t, i = super().step(action)
        return self.wrap_observation(s), r, t, i

    def reset(self, **kwargs):
        return self.wrap_observation(super().reset())

    def wrap_observation(self, observation):
        obs, task = observation
        _, h, w = obs.shape
        env = self.env.unwrapped

        # subtask pointer
        task_type, task_count, task_object_type = env.subtask

        task_type_one_hot = np.zeros((len(env.task_types), h, w), dtype=bool)
        task_count_one_hot = np.zeros((env.max_task_count, h, w), dtype=bool)
        task_object_one_hot = np.zeros((len(env.object_types), h, w), dtype=bool)

        task_type_one_hot[task_type, :, :] = True
        task_count_one_hot[task_count - 1, :, :] = True
        task_object_one_hot[task_object_type, :, :] = True

        # task spec
        def task_iterator():
            for column in env.task.T:  # transpose for easy splitting in Subtasks module
                for word in column:
                    yield word

        task_spec = np.zeros(((3 * env.n_subtasks), h, w), dtype=int)
        for row, word in zip(task_spec, task_iterator()):
            row[:] = word

        # task_objects_one_hot = np.zeros((h, w), dtype=bool)
        # idx = [k for k, v in env.objects.items() if v == task_object_type]
        # set_index(task_objects_one_hot, idx, True)

        next_subtask = np.full((1, h, w), env.next_subtask)

        stack = np.vstack(
            [
                obs,
                task_type_one_hot,
                task_count_one_hot,
                task_object_one_hot,
                task_spec,
                next_subtask,
            ]
        )
        # print('obs', obs.shape)
        # print('task_type', task_type_one_hot.shape)
        # print('task_objects', task_objects_one_hot.shape)
        # print('task_spec', task_spec.shape)
        # print('iterate', iterate.shape)
        # print('stack', stack.shape)

        # names = ['obstacles'] + list(env.object_types) + ['ice', 'agent'] + \
        #         list(env.task_types) + ['task objects']
        # assert len(obs) == len(names)
        # for array, name in zip(obs, names):
        #     print(name)
        #     print(array)

        return stack.astype(float)

    def render(self, mode="human"):
        super().render(mode=mode)
        if self.last_g is not None:
            g_type, g_count, g_obj = tuple(self.task[self.last_g])
            env = self.env.unwrapped
            print(
                "Assigned subtask:",
                env.task_types[g_type],
                g_count,
                env.object_types[g_obj],
            )
        input("paused")


def get_subtasks_obs_sections(task_space):
    n_subtasks, size_subtask = task_space.shape
    return Obs(
        base=(
            1 + task_space.nvec[0, 2] + 1 + 1  # obstacles  # objects one hot  # ice
        ),  # agent
        subtask=(sum(task_space.nvec[0])),  # one hots
        task=size_subtask * n_subtasks,  # int codes
        next_subtask=1,
    )


def get_subtasks_action_sections(action_spaces):
    return SubtasksActions(
        *[s.shape[0] if isinstance(s, Box) else 1 for s in action_spaces]
    )
