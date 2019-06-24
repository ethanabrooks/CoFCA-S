from collections import namedtuple

from gym import spaces
import numpy as np

import ppo.subtasks.wrappers

Obs = namedtuple('Obs', 'base subtask subtasks control next_subtask')


class Wrapper(ppo.subtasks.wrappers.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_spaces = ppo.subtasks.wrappers.Obs(*self.observation_space.spaces)
        interactions, max_count, objects = obs_spaces.subtask.nvec
        n_subtasks = 2 * env.n_subtasks  # 2 per branch
        control_nvec = np.array([[objects, n_subtasks, n_subtasks]]).repeat(
            env.n_subtasks, axis=0)
        subtasks_nvec = np.expand_dims(obs_spaces.subtask.nvec, 0).repeat(
            n_subtasks, axis=0)
        obs = Obs(
            base=obs_spaces.base,
            subtask=obs_spaces.subtask,
            subtasks=spaces.MultiDiscrete(subtasks_nvec),
            next_subtask=obs_spaces.next_subtask,
            control=spaces.MultiDiscrete(control_nvec))
        self.observation_space = spaces.Tuple(obs)
        self.subtasks = None

    def wrap_observation(self, observation):
        obs, task = observation
        _, h, w = obs.shape
        env = self.env.unwrapped

        def get_subtasks():
            for branch in task:
                yield branch.true_path
                yield branch.false_path

        self.subtasks = list(get_subtasks())

        def get_control_flow():
            for branch in task:
                yield (
                    branch.condition,
                    self.subtasks.index(branch.true_path),
                    self.subtasks.index(branch.false_path),
                )

        control = list(get_control_flow())

        observation = Obs(
            base=obs,
            subtask=env.subtask,
            subtasks=self.subtasks,
            control=control,
            next_subtask=[env.next_subtask],
        )
        return np.concatenate(
            [np.array(list(x)).flatten() for x in observation])

    def chosen_subtask(self, env):
        return self.subtasks[self.last_g]
