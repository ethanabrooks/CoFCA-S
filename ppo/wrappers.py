import gym
from gym import spaces
from gym.spaces import Box
import numpy as np
import torch

from common.vec_env import VecEnvWrapper
from common.vec_env.vec_normalize import VecNormalize as VecNormalize_
from ppo.utils import set_index
from rl_utils import onehot


class SubtasksWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_space, task_space = env.observation_space.spaces
        assert np.all(task_space.nvec == task_space.nvec[0])
        task_channels = task_space.nvec[0, 0]
        obs_shape = np.array(obs_space.nvec.shape)
        obs_shape[0] += task_channels  # for task type one hot
        obs_shape[0] += np.prod(
            task_space.nvec.shape)  # for task specification
        obs_shape[0] += 1  # for task_objects
        obs_shape[0] += 1  # for iterate
        self.observation_space = Box(0, 1, shape=obs_shape)

    def observation(self, observation):
        obs, task = observation
        _, h, w = obs.shape
        env = self.env.unwrapped

        # subtask pointer
        task_type, task_count, task_object_type = env.subtask
        task_type_one_hot = np.zeros((len(env.task_types), h, w), dtype=bool)
        task_type_one_hot[task_type, :, :] = True

        # task spec
        def task_iterator():
            for column in env.task.T:  # transpose for easy splitting in Subtasks module
                for word in column:
                    yield word

        task_spec = np.zeros(((3 * env.n_subtasks), h, w), dtype=int)
        for row, word in zip(task_spec, task_iterator()):
            row[:] = word

        # TODO: make this less easy
        task_objects_one_hot = np.zeros((h, w), dtype=bool)
        idx = [k for k, v in env.objects.items() if v == task_object_type]
        set_index(task_objects_one_hot, idx, True)

        iterate = np.full((1, h, w), env.iterate)

        stack = np.vstack([
            obs, task_type_one_hot,
            np.expand_dims(task_objects_one_hot, 0), task_spec, iterate
        ])

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
