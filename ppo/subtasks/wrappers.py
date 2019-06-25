from collections import namedtuple

import gym
from gym import spaces
from gym.spaces import Box
import numpy as np

Actions = namedtuple('Actions', 'a cr cg g')
Obs = namedtuple('Obs', 'base subtask task next_subtask')


class DebugWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_guess = None
        self.last_reward = None
        self.subtask_space = env.task_space.nvec[0]

    def step(self, action):
        action_sections = Wrapper.parse_action(self, action)
        actions = Actions(*[
            int(x.item()) for x in np.split(action,
                                            np.cumsum(action_sections)[:-1])
        ])
        s, _, t, i = super().step(action)
        guess = int(actions.g)
        truth = int(self.env.unwrapped.subtask_idx)
        r = float(np.all(guess == truth)) - 1
        self.last_guess = guess
        self.last_reward = r
        return s, r, t, i

    def render(self, mode='human'):
        print('########################################')
        super().render(sleep_time=0)
        print('guess', self.last_guess)
        print('truth', self.env.unwrapped.subtask_idx)
        print('reward', self.last_reward)
        # input('pause')


class Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_space, task_space = env.observation_space.spaces
        assert np.all(task_space.nvec == task_space.nvec[0])
        self.task_space = task_space
        self.observation_space = spaces.Tuple(
            Obs(
                base=Box(0, 1, shape=obs_space.nvec),
                subtask=spaces.Discrete(task_space.nvec.shape[0]),
                task=task_space,
                next_subtask=spaces.Discrete(2),
            ))
        self.action_space = spaces.Tuple(
            Actions(
                a=env.action_space,
                g=spaces.Discrete(env.n_subtasks),
                cg=spaces.Discrete(2),
                cr=spaces.Discrete(2),
            ))
        self.last_g = None

    def step(self, action):
        actions = Actions(*np.split(action, len(self.action_space.spaces)))
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
        observation = Obs(base=obs,
                          subtask=env.subtask_idx,
                          task=env.task,
                          next_subtask=env.next_subtask)
        # for obs, space in zip(observation, self.observation_space.spaces):
        # assert space.contains(np.array(obs))
        return np.concatenate([np.array(x).flatten() for x in observation])

    def render(self, mode='human', **kwargs):
        super().render(mode=mode)
        if self.last_g is not None:
            env = self.env.unwrapped
            g_type, g_count, g_obj = tuple(self.chosen_subtask(env))
            print(
                'Assigned subtask:',
                env.interactions[g_type],
                g_count + 1,
                env.object_types[g_obj],
            )
        input('paused')

    def chosen_subtask(self, env):
        return env.task[self.last_g]
