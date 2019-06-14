from collections import namedtuple

import gym
import numpy as np
import torch
from gym import spaces
from gym.spaces import Box

from common.vec_env import VecEnvWrapper
from common.vec_env.vec_normalize import VecNormalize as VecNormalize_
from gridworld_env.subtasks_gridworld import ObsSections
from rl_utils import onehot

SubtasksActions = namedtuple('SubtasksActions', 'a b g_embed c l g_int')


def get_subtasks_obs_sections(task_space):
    n_subtasks, size_subtask = task_space.shape
    return ObsSections(
        base=(
            1 +  # obstacles
            task_space.nvec[0, 2] +  # objects one hot
            1 +  # ice
            1),  # agent
        subtask=(sum(task_space.nvec[0])),  # one hots
        task=size_subtask * n_subtasks,  # int codes
        next_subtask=1)


def get_subtasks_action_sections(action_spaces):
    return SubtasksActions(
        *[s.shape[0] if isinstance(s, Box) else 1 for s in action_spaces])


class DebugWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # self.action_space = spaces.Discrete(
        # int(self.size_subtask_space * self.size_action_space))
        self.possible_subtasks = np.array([
            [0, 1, 0],
            [0, 1, 1],
        ])
        self.last_action = None
        self.last_reward = None

    def step(self, action):
        action_sections = get_subtasks_action_sections(self.action_space)
        actions = SubtasksActions(*np.split(action, action_sections))
        # action, subtask = np.unravel_index(
        # action, (self.size_action_space, self.size_subtask_space))
        s, _, t, i = super().step(action)
        guess = self.possible_subtasks[int(actions.g)]
        truth = self.env.unwrapped.subtask
        r = float(np.all(guess == truth))
        self.last_action = actions
        self.last_reward = r
        return s, r, t, i  # TODO

    def render(self, mode='human'):
        action = self.last_action
        print('########################################')
        super().render()
        if action is not None:
            g = int(action.g)
            print('guess', g, self.possible_subtasks[g])
        print('truth', self.env.unwrapped.subtask)
        print('reward', self.last_reward)


class SubtasksWrapper(gym.Wrapper):
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
                b=spaces.Discrete(2),
                g_embed=spaces.Box(
                    low=0, high=1, shape=(task_space.nvec[0].sum(), )),
                g_int=spaces.Discrete(task_space.nvec[0].prod()),
                c=spaces.Discrete(2),
                l=spaces.Discrete(3)))

    def step(self, action):
        action_sections = np.cumsum(
            get_subtasks_action_sections(
                self.action_space.spaces))[:-1].astype(int)
        actions = SubtasksActions(*np.split(action, action_sections))
        action = int(actions.a)
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
        task_object_one_hot = np.zeros((len(env.object_types), h, w),
                                       dtype=bool)

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

        stack = np.vstack([
            obs, task_type_one_hot, task_count_one_hot, task_object_one_hot,
            task_spec, next_subtask
        ])
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


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:0] = 0
        return observation


class AddTimestep(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AddTimestep, self).__init__(env)
        self.observation_space = Box(
            self.observation_space.low[0],
            self.observation_space.high[0],
            [self.observation_space.shape[0] + 1],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return np.concatenate((observation, [self.env._elapsed_steps]))


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = 'cpu'
        # TODO: Fix data types

    @staticmethod
    def extract_numpy(obs):
        if not isinstance(obs, (list, tuple)):
            return obs
        assert len(obs) == 1
        return obs[0]

    def reset(self):
        obs = self.extract_numpy(self.venv.reset())
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        actions = actions.squeeze(1).cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = self.extract_numpy(obs)
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).float()
        return obs, reward, done, info

    def to(self, device):
        self.device = device
        self.venv.to(device)


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs):
        if self.ob_rms:
            if self.training:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) /
                          np.sqrt(self.ob_rms.var + self.epsilon),
                          -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        self.stacked_obs = torch.zeros((venv.num_envs, ) + low.shape)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:]
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()

    def to(self, device):
        self.stacked_obs = self.stacked_obs.to(device)
        self.venv.to(device)


class OneHotWrapper(gym.Wrapper):
    def wrap_observation(self, obs, observation_space=None):
        if observation_space is None:
            observation_space = self.observation_space
        if isinstance(observation_space, spaces.Discrete):
            return onehot(obs, observation_space.n)
        if isinstance(observation_space, spaces.MultiDiscrete):
            assert observation_space.contains(obs)

            def one_hots():
                nvec = observation_space.nvec
                for o, n in zip(
                        obs.reshape(len(obs), -1).T,
                        nvec.reshape(len(nvec), -1).T):
                    yield onehot(o, n)

            return np.concatenate(list(one_hots()), axis=-1)


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None
