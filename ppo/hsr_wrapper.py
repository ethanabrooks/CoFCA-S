# third party
import numpy as np
from collections import namedtuple
from multiprocessing import Pipe, Process

# first party
import torch
from baselines.common.vec_env import CloudpickleWrapper, VecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from environments import hsr

from utils.utils import concat_spaces, space_shape, vectorize, unwrap_env


class HSREnv(hsr.HSREnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Sadly, ppo code really likes boxes, so had to concatenate things
        self.observation_space = concat_spaces(
            self.observation_space.spaces, axis=0)

    def step(self, action):
        s, r, t, i = super().step(action)
        return vectorize(s), r, t, i

    def reset(self):
        return vectorize(super().reset())


class MoveGripperEnv(HSREnv, hsr.MoveGripperEnv):
    pass


StepData = namedtuple('StepData', 'actions reward_params')


# TODO: test with multiple envs
# TODO: test with small nsteps

class Observation(namedtuple('Observation', 'observation achieved params')):
    def replace(self, *args, **kwargs):
        return self._replace(*args, **kwargs)


class RewardStructure:
    def __init__(self, num_processes, subspace_sizes, reward_function):
        self.reward_function = reward_function
        self.function = reward_function
        # TODO make sure we are doing this correctly
        self.subspace_sizes = Observation(*subspace_sizes)
        starts = _, *ends = np.cumsum([0] + subspace_sizes)
        self.subspace_slices = Observation(*[slice(*s) for s in zip(starts, ends)])
        self.reward_params = torch.zeros(
            num_processes,
            self.subspace_sizes.params,
            requires_grad=True,
        )


class UnsupervisedEnv(hsr.HSREnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        old_spaces = hsr.Observation(*self.observation_space.spaces)
        spaces = Observation(
            observation=old_spaces.observation,
            params=old_spaces.goal,
            achieved=old_spaces.goal)

        self.subspace_sizes = [space_shape(space)[0] for space in spaces]

        # space of observation needs to exclude reward param
        self.observation_space = concat_spaces(spaces, axis=0)
        self.reward_params = self.achieved_goal()

    @staticmethod
    def reward_function(achieved, params, dim):
        return -((achieved - params) ** 2).sum(dim)

    def compute_reward(self):
        return self.reward_function(achieved=self.achieved_goal(),
                                    params=self.reward_params, dim=0)

    def step(self, actions):
        s, r, t, i = super().step(actions)
        observation = Observation(observation=s.observation, params=s.goal,
                                  achieved=self.achieved_goal())
        return vectorize(observation), r, t, i

    def reset(self):
        s = super().reset()
        return vectorize(Observation(observation=s.observation, params=s.goal,
                                     achieved=self.achieved_goal()))

    def compute_terminal(self):
        return False

    def achieved_goal(self):
        return self.gripper_pos()

    def new_goal(self):
        return self.reward_params

    def set_reward_params(self, param):
        self.reward_params = param
        self.set_goal(param)


def unwrap_unsupervised(env):
    return unwrap_env(env, lambda e: hasattr(e, 'set_reward_params'))


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'set_reward_params':
            unwrap_unsupervised(env).set_reward_params(data)
        else:
            raise NotImplementedError


class UnsupervisedSubprocVecEnv(SubprocVecEnv):
    # noinspection PyMissingConstructor
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [
            Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in
            zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things
            # to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def set_reward_params(self, params):
        for remote, param in zip(self.remotes, params):
            remote.send(('set_reward_params', param))


class UnsupervisedDummyVecEnv(DummyVecEnv):
    def set_reward_params(self, params):
        for env, param in zip(self.envs, params):
            unwrap_unsupervised(env).set_reward_params(param)
