# third party
from collections import namedtuple
from multiprocessing import Pipe, Process

import numpy as np
# first party
from baselines.common.vec_env import CloudpickleWrapper, VecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from environments import hsr
from gym.spaces import Box
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


class Observation(namedtuple('Observation', 'observation achieved params')):
    def replace(self, *args, **kwargs):
        return self._replace(*args, **kwargs)


class UnsupervisedEnv(hsr.HSREnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        old_spaces = hsr.Observation(*self.observation_space.spaces)
        self.goals = []
        spaces = Observation(
            observation=old_spaces.observation,
            params=old_spaces.goal,
            achieved=old_spaces.goal)

        # subspace_sizes used for splitting concatenated tensor observations
        self.subspace_sizes = [space_shape(space)[0] for space in spaces]
        for n in self.subspace_sizes:
            assert isinstance(n, int)

        # space of observation needs to exclude reward param
        self.observation_space = concat_spaces(spaces, axis=0)
        self.reward_params = self.achieved_goal()

    def step(self, actions):
        s, r, t, i = super().step(actions)
        observation = Observation(observation=s.observation, params=s.goal,
                                  achieved=self.achieved_goal())
        return vectorize(observation), r, t, i

    def reset(self):
        o = super().reset()
        print('reset params', o.goal)
        return vectorize(Observation(observation=o.observation, params=o.goal,
                                     achieved=self.achieved_goal()))

    def store_goal(self, goal):
        self.goals.append(goal)

    def achieved_goal(self):
        return self.gripper_pos()

    def new_goal(self):
        return self.reward_params


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
        elif cmd == 'store_goal':
            unwrap_unsupervised(env).store_goal(data)
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

    def store_goals(self, goals):
        for remote, goal in zip(self.remotes, goals):
            remote.send(('store_goal', goal))


class UnsupervisedDummyVecEnv(DummyVecEnv):
    def store_goals(self, goals):
        for env, goal in zip(self.envs, goals):
            print('sent params', goal)
            unwrap_unsupervised(env).set_goal(goal)
