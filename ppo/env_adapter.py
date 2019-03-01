# third party
from multiprocessing import Pipe, Process

from gym.spaces import Box, Discrete
import numpy as np

# first party
from common.vec_env import CloudpickleWrapper, VecEnv
from common.vec_env.dummy_vec_env import DummyVecEnv
from common.vec_env.subproc_vec_env import SubprocVecEnv
import gridworld_env
import hsr
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


class TasksHSREnv(hsr.env.HSREnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        old_spaces = hsr.env.Observation(*self.observation_space.spaces)
        spaces = Observation(
            observation=old_spaces.observation, task=old_spaces.task)

        # subspace_sizes used for splitting concatenated tensor observations
        self._subspace_sizes = Observation(
            *[space_shape(space)[0] for space in spaces])
        for n in self.subspace_sizes:
            assert isinstance(n, int)

        # space of observation needs to exclude reward param
        self.observation_space = concat_spaces(spaces)
        self.reward_params = self.achieved_task()

    @property
    def subspace_sizes(self):
        return self._subspace_sizes

    def step(self, actions):
        s, r, t, i = super().step(actions)
        i.update(task=self.task)
        observation = Observation(observation=s.observation, task=s.task)
        return vectorize(observation), r, t, i

    def reset(self):
        o = super().reset()
        return vectorize(Observation(observation=o.observation, task=o.task))

    def new_task(self):
        return self.task


class TasksMoveGripperEnv(TasksHSREnv, hsr.env.MoveGripperEnv):
    pass


class GridWorld(gridworld_env.gridworld.GridWorld):
    def __init__(self, random=None, *args, **kwargs):
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
    def __init__(self, random=None, *args, **kwargs):
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


class TasksGridWorld(GridWorld):
    def __init__(self, no_task_in_obs, task_letter='*', *args, **kwargs):
        self.include_task_in_obs = not no_task_in_obs
        super().__init__(*args, **kwargs)
        self.task_states = np.ravel_multi_index(
            np.where(
                np.logical_not(
                    np.logical_or(
                        np.isin(self.desc, self.blocked),
                        np.isin(self.desc, self.terminal),
                    ))),
            dims=self.desc.shape)
        self.observation_size = space_to_size(self.observation_space)
        self.evaluation = False

        # task stuff
        self.task_index = None
        self.task_vector = None
        self.task_space = Discrete(self.task_states.size)
        self.task_letter = task_letter
        self.num_tasks = len(self.task_states)
        self.task_dist = np.ones_like(self.task_states) / self.num_tasks

        size = self.observation_size
        if self.include_task_in_obs:
            size *= 2
        self.observation_space = Box(
            low=np.zeros(size),
            high=np.ones(size),
        )

    def set_task(self, task_index):
        self.task_index = task_index
        task_state = self.task_states[self.task_index]
        self.task_vector = onehot(task_state, self.observation_size)
        self.assign(**{self.task_letter: [task_state]})

    def reset(self):
        if not self.evaluation:
            self.set_task(np.random.choice(self.num_tasks, p=self.task_dist))
        return super().reset()

    def obs_vector(self, obs):
        components = [onehot(obs, self.observation_size)]
        if self.include_task_in_obs:
            components.append(self.task_vector)
        return vectorize(components)


class TrainTasksGridWorld(TasksGridWorld):
    def set_task_dist(self, dist):
        self.task_dist = dist


def unwrap_tasks(env):
    return unwrap_env(env, lambda e: hasattr(e, 'set_task_dist'))


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
        elif cmd == 'set_task_dist':
            unwrap_tasks(env).set_task_dist(data)
        elif cmd == 'get_task':
            remote.send(unwrap_tasks(env).task_index)
        else:
            raise NotImplementedError


class TasksSubprocVecEnv(SubprocVecEnv):
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

    def set_task_dist(self, i, dist):
        self.remotes[i].send(('set_task_dist', dist))

    def get_tasks(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_task', None))
        self.waiting = True
        tasks = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return tasks


class TasksDummyVecEnv(DummyVecEnv):
    def set_task_dist(self, i, dist):
        unwrap_tasks(self.envs[i]).set_task_dist(dist)

    def get_tasks(self):
        return [unwrap_tasks(env).task_index for env in self.envs]
