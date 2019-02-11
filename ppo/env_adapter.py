# third party

from multiprocessing import Pipe, Process

import numpy as np
from gym.spaces import Box

import gridworld_env
import hsr
# first party
from common.vec_env import CloudpickleWrapper, VecEnv
from common.vec_env.dummy_vec_env import DummyVecEnv
from common.vec_env.subproc_vec_env import SubprocVecEnv
from hsr.env import Observation
from utils.gym import concat_spaces, space_shape, space_to_size, unwrap_env
from utils.numpy import onehot, vectorize


class HSREnv(hsr.env.HSREnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Sadly, ppo code really likes boxes, so had to concatenate things
        self.observation_space = concat_spaces(self.observation_space.spaces)

    def step(self, action):
        s, r, t, i = super().step(action)
        return vectorize(s), r, t, i

    def reset(self):
        return vectorize(super().reset())


class MoveGripperEnv(HSREnv, hsr.env.MoveGripperEnv):
    pass


class UnsupervisedHSREnv(hsr.env.HSREnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        old_spaces = hsr.env.Observation(*self.observation_space.spaces)
        spaces = Observation(
            observation=old_spaces.observation, goal=old_spaces.goal)

        # subspace_sizes used for splitting concatenated tensor observations
        self._subspace_sizes = Observation(
            *[space_shape(space)[0] for space in spaces])
        for n in self.subspace_sizes:
            assert isinstance(n, int)

        # space of observation needs to exclude reward param
        import ipdb
        ipdb.set_trace()
        self.observation_space = concat_spaces(spaces)
        self.reward_params = self.achieved_goal()

    @property
    def subspace_sizes(self):
        return self._subspace_sizes

    def step(self, actions):
        s, r, t, i = super().step(actions)
        i.update(goal=self.goal)
        observation = Observation(observation=s.observation, goal=s.goal)
        return vectorize(observation), r, t, i

    def reset(self):
        o = super().reset()
        return vectorize(Observation(observation=o.observation, goal=o.goal))

    def new_goal(self):
        return self.goal


class UnsupervisedMoveGripperEnv(UnsupervisedHSREnv, hsr.env.MoveGripperEnv):
    pass


class GridWorld(gridworld_env.gridworld.GridWorld):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = Box(
            low=np.zeros(self.observation_space.n),
            high=np.ones(self.observation_space.n),
        )
        self.observation_size = space_to_size(self.observation_space)

    def obs_vector(self, obs):
        return onehot(obs, self.observation_size)

    def step(self, actions):
        s, r, t, i = super().step(actions)
        return self.obs_vector(s), r, t, i

    def reset(self):
        o = super().reset()
        return self.obs_vector(o)


class RandomGridWorld(gridworld_env.random_gridworld.RandomGridWorld):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_sizes = [
            space_to_size(space) for space in self.observation_space.spaces
        ]
        self.observation_space = Box(
            low=np.zeros(sum(self.observation_sizes)),
            high=np.zeros(sum(self.observation_sizes)))

    def obs_vector(self, obs):
        return vectorize(
            [onehot(x, size) for x, size in zip(obs, self.observation_sizes)])

    def step(self, actions):
        s, r, t, i = super().step(actions)
        return self.obs_vector(s), r, t, i

    def reset(self):
        o = super().reset()
        return self.obs_vector(o)


class UnsupervisedGridWorld(GridWorld):
    def __init__(self, *args, random=None, goal_letter='*', **kwargs):
        self.goal = None
        self.goal_letter = goal_letter
        super().__init__(*args, **kwargs)
        self.goal_states = np.ravel_multi_index(
            np.where(np.logical_not(np.isin(self.desc, self.blocked))),
            dims=self.desc.shape)
        self.observation_size = space_to_size(self.observation_space)
        self.goal_space = Discrete(self.goal_states.size)
        self.observation_space = Box(
            low=np.zeros(self.observation_size * 2),
            high=np.ones(self.observation_size * 2),
        )

    def set_goal(self, goal_index):
        goal_state = self.goal_states[goal_index]
        self.assign(**{self.goal_letter: [goal_state]})
        self.goal = onehot(goal_state, self.observation_size)
        assert self.desc[self.decode(goal_state)] == self.goal_letter

    def obs_vector(self, obs):
        return vectorize([onehot(obs, self.observation_size), self.goal])


def unwrap_unsupervised(env):
    return unwrap_env(env, lambda e: hasattr(e, 'set_goal'))


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
        elif cmd == 'set_goal':
            unwrap_unsupervised(env).set_goal(data)
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
            Process(
                target=worker,
                args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote,
                 env_fn) in zip(self.work_remotes, self.remotes, env_fns)
        ]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things
            # to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def set_goal(self, goal, i):
        self.remotes[i].send(('set_goal', goal))


class UnsupervisedDummyVecEnv(DummyVecEnv):
    def set_goal(self, goal, i):
        env = unwrap_unsupervised(self.envs[i])
        env.set_goal(goal)
