from collections import namedtuple

import numpy as np
from gym import spaces

import ppo.subtasks.wrappers

Obs = namedtuple('Obs', 'base subtask task control next_subtask')


class Wrapper(ppo.subtasks.wrappers.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_spaces = ppo.subtasks.wrappers.Obs(*self.observation_space.spaces)
        interactions, max_count, objects = obs_spaces.subtask.nvec
        n_subtasks = 2 * env.n_subtasks  # 2 per branch
        control_nvec = np.array([[objects, n_subtasks, n_subtasks]]).repeat(
            env.n_subtasks, axis=0)
        self.observation_space = spaces.Tuple(
            Obs(base=obs_spaces.base,
                subtask=obs_spaces.subtask,
                task=obs_spaces.task,
                next_subtask=obs_spaces.next_subtask,
                control=spaces.MultiDiscrete(control_nvec)))

    def wrap_observation(self, observation):
        obs, task = observation
        _, h, w = obs.shape
        env = self.env.unwrapped

        def get_subtasks():
            for branch in task:
                yield branch.true_path
                yield branch.false_path

        subtasks = list(get_subtasks())

        def get_control_flow():
            for branch in task:
                yield (
                    branch.condition,
                    subtasks.index(branch.true_path),
                    subtasks.index(branch.false_path),
                )

        control = list(get_control_flow())

        observation = Obs(
            base=obs,
            subtask=env.subtask,
            task=env.task,
            control=control,
            next_subtask=env.next_subtask,
        )
        return np.concatenate([np.array(x).flatten() for x in observation])
