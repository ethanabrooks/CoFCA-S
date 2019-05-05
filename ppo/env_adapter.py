# third party
from multiprocessing import Pipe, Process
from pathlib import Path
import pickle
import time
from typing import List

# first party
from gym.spaces import Box, Discrete
import numpy as np

from common.vec_env import CloudpickleWrapper, VecEnv
from common.vec_env.dummy_vec_env import DummyVecEnv
from common.vec_env.subproc_vec_env import SubprocVecEnv
import gridworld_env
import hsr
from hsr.env import GoalSpec
from mujoco_py import MjSimState
from rl_utils.gym import space_to_size, unwrap_env
from rl_utils.numpy import onehot, vectorize


class HSREnv(hsr.env.HSREnv):
    def __init__(self, block_space, goal_space, geofence, min_lift_height,
                 image_dims, record_separate_episodes, **kwargs):
        self.evaluation = False
        self.num_eval = 1
        goal = goal_space
        if block_space is None:
            starts = dict()
            goal = GoalSpec(
                a=lambda: self.gripper_pos(), b=goal, distance=geofence)
        else:
            default = np.zeros(7)  # x y z q1 q2 q3 q4
            default[2] = .418
            low = default.copy()
            high = default.copy()
            low[[0, 1, 3, 6]] = block_space.low
            high[[0, 1, 3, 6]] = block_space.high
            starts = dict(block0joint=Box(low=low, high=high))
            if min_lift_height:
                goal = np.zeros(3)
                goal[2] += .418 + min_lift_height
            goal = GoalSpec(a='block0', b=goal, distance=geofence)

        super().__init__(starts=starts, goals=[goal], **kwargs)


class SaveStateHSREnv(HSREnv):
    def __init__(self, save_path: Path, **kwargs):
        super().__init__(**kwargs)
        self.save_path = save_path
        self.saved_state = []
        self.reset_once = False

    def close(self):
        with self.save_path.open('wb') as f:
            pickle.dump(list(zip(*self.saved_state)), f)
        super().close()

    def step(self, action):
        s, r, t, i = super().step(action)
        self.saved_state.append((self.sim.get_state(),
                                 self.render('rgb_array')))
        return s, r, t, i

    def reset(self):
        if self.reset_once:
            self.close()
            exit()
        s = super().reset()
        self.reset_once = True
        self.saved_state.append((self.sim.get_state(),
                                 self.render('rgb_array')))
        return s


class AutoCurriculumHSREnv(HSREnv):
    def __init__(self, start_states: List[MjSimState],
                 start_images: List[np.ndarray], random_initial_steps: int,
                 **kwargs):
        self.random_initial_steps = random_initial_steps
        self.start_images = start_images
        self.start_states = start_states
        self.num_tasks = len(start_states)
        self.task_space = Discrete(len(start_states))
        self.task_dist = np.ones(len(start_states)) / len(start_states)
        self.task_prob = None
        self.task_index = None
        self._time_steps = 0
        super().__init__(**kwargs)
        self.num_eval = self.num_tasks

    def new_state(self):
        if self.evaluation:
            return self.start_states[self.task_index]
        self.task_index = self.np_random.choice(
            len(self.start_states), p=self.task_dist)
        self.task_prob = self.task_dist[self.task_index]
        return self.start_states[self.task_index]

    def reset(self):
        o = super().reset()
        if self.evaluation:
            return o
        o, r, t, i = self.step(
            self.action_space.sample(), steps=self.random_initial_steps)
        if not t:
            return o
        return self.reset()  # stepped into terminal state. try again

    def get_task_and_prob(self):
        return self.task_index, self.task_prob

    def set_task(self, task_index):
        self.task_index = task_index

    def set_task_dist(self, dist):
        self.task_dist = dist

    def get_start_xpos(self):
        xpos = []
        for state in self.start_states:
            self.sim.set_state(state)
            self.sim.forward()
            xpos.append((self.gripper_pos(), self.block_pos()))
        self.reset()
        return xpos


class GridWorld(gridworld_env.gridworld.GridWorld):
    def __init__(self, env_id, render: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.evaluation = False

        env_id = env_id[:-len('GridWorld-v0')]
        if env_id in ['8x8Wall', '12x3Wall', '16x16Wall']:
            self.include_task_in_obs = True
        elif env_id in ['Cliff', 'Shortcut', 'TwoPaths']:
            self.include_task_in_obs = False
        elif env_id in ['5x13Lava']:
            self.include_task_in_obs = False
            self.assign(**{"*": [40]})
        else:
            raise RuntimeError('Invalid ID:', env_id)

        self._render = render
        self.observation_space = Box(
            low=np.zeros(self.observation_space.n),
            high=np.ones(self.observation_space.n),
        )
        self.observation_size = space_to_size(self.observation_space)

    def eval_mode(self, rank=None):
        self.evaluation = True
        self._render = self._render and rank == 0

    def obs_vector(self, obs):
        return onehot(obs, self.observation_size)

    def step(self, actions):
        if self._render:
            self.render()
            time.sleep(.2)
        s, r, t, i = super().step(actions)
        if t and self._render:
            self.render()
            time.sleep(2)
        return self.obs_vector(s), r, t, i

    def reset(self):
        o = super().reset()
        return self.obs_vector(o)


class TasksGridWorld(GridWorld):
    def __init__(self, env_id: str, task_letter='*', *args, **kwargs):
        super().__init__(*args, env_id=env_id, **kwargs)
        self.task_states = np.ravel_multi_index(
            np.where(self.desc == ' '), dims=self.desc.shape)
        self.observation_size = space_to_size(self.observation_space)

        # task stuff
        self.task_index = None
        self.task_vector = None
        self.task_space = Discrete(self.task_states.size)
        self.task_letter = task_letter
        self.num_tasks = len(self.task_states)
        self.task_dist = np.ones_like(self.task_states) / self.num_tasks
        self.task_prob = 1 / self.num_tasks

        env_id = env_id[:-len('GridWorld-v0')]

        if env_id in ['8x8Wall', '16x16Wall', '5x13Lava']:
            self.num_eval = self.num_tasks
        elif env_id in ['12x3Wall']:
            self.num_eval = 9
        elif env_id in ['Cliff', 'Shortcut', 'TwoPaths']:
            self.num_eval = 1
            assert self.decode(self.task_states[0]) == (0, 1)
        else:
            raise RuntimeError('Invalid ID:', env_id)

        size = self.observation_size
        if self.include_task_in_obs:
            size *= 2
        self.observation_space = Box(
            low=np.zeros(size),
            high=np.ones(size),
        )

    def get_task_and_prob(self):
        return self.task_index, self.task_prob

    def set_task(self, task_index):
        self.task_index = task_index
        task_state = self.task_states[self.task_index]
        self.task_vector = onehot(task_state, self.observation_size)
        self.assign(**{self.task_letter: [task_state]})

    def obs_vector(self, obs):
        components = [onehot(obs, self.observation_size)]
        if self.include_task_in_obs:
            components.append(self.task_vector)
        return vectorize(components)


class RMaxGridWorld(TasksGridWorld):
    def __init__(self, visits_until_known, env_id, **kwargs):
        super().__init__(env_id=env_id, **kwargs)

        self.task_states = np.ravel_multi_index(
            np.where(self.desc == ' '), dims=self.desc.shape)
        self.task_index = 0
        env_id = env_id[:-len('GridWorld-v0')]
        if env_id in ['8x8Wall', '12x3Wall', '16x16Wall']:
            self.rmax = 1
        elif env_id in ['5x13Lava']:
            self.rmax = 100
            self.num_eval = len(self.task_states)
        else:
            raise RuntimeError('Invalid ID:', env_id)

        self.visits_until_known = visits_until_known
        self.visit_count = np.zeros((self.observation_size,
                                     self.action_space.n))

    def set_task(self, task_index):
        self.task_index = task_index
        task_state = self.task_states[self.task_index]
        self.assign(**{'*': [task_state]})

    def step(self, actions):
        self.visit_count[self.s, actions] += 1
        visit_count = self.visit_count[self.s, actions]
        s, r, t, i = super().step(actions)
        if visit_count < self.visits_until_known and not self.evaluation:
            r = self.rmax
            t = True
        return s, r, t, i


class TrainTasksGridWorld(TasksGridWorld):
    def set_task_dist(self, dist):
        self.task_dist = dist

    def reset(self):
        if not self.evaluation:
            task_index = self.np_random.choice(
                self.num_tasks, p=self.task_dist)
            self.set_task(task_index)
            self.task_prob = self.task_dist[task_index]
        return super().reset()


def unwrap_tasks(env):
    return unwrap_env(env, lambda e: hasattr(e, 'get_task_and_prob'))


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
        elif cmd == 'get_task_and_prob':
            remote.send(unwrap_tasks(env).get_task_and_prob())
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

    def get_tasks_and_probs(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_task_and_prob', None))
        self.waiting = True
        tasks, probs = zip(*[remote.recv() for remote in self.remotes])
        self.waiting = False
        return tasks, probs


class TasksDummyVecEnv(DummyVecEnv):
    def set_task_dist(self, i, dist):
        unwrap_tasks(self.envs[i]).set_task_dist(dist)

    def get_tasks_and_probs(self):
        return zip(
            *[unwrap_tasks(env).get_task_and_prob() for env in self.envs])
