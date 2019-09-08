import gym
from gym import spaces
from gym.spaces import Box
import numpy as np
import torch

from common.vec_env import VecEnvWrapper
from common.vec_env.vec_normalize import VecNormalize as VecNormalize_
from rl_utils import onehot

SubtasksActions = namedtuple('SubtasksActions', 'a cr cg g')
SubtasksObs = namedtuple('SubtasksObs', 'base subtask task next_subtask')


class DebugWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_guess = None
        self.last_reward = None
        self.subtask_space = env.task_space.nvec[0]

    def step(self, action):
        action_sections = SubtasksWrapper.parse_action(self, action)
        actions = SubtasksActions(*[
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


class SubtasksWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_space, task_space = env.observation_space.spaces
        assert np.all(task_space.nvec == task_space.nvec[0])
        self.task_space = task_space
        self.observation_space = spaces.Tuple(
            SubtasksObs(
                base=Box(0, 1, shape=obs_space.nvec),
                subtask=spaces.MultiDiscrete(task_space.nvec[0]),
                task=task_space,
                next_subtask=spaces.Discrete(2),
            ))
        self.action_space = spaces.Tuple(
            SubtasksActions(
                a=env.action_space,
                g=spaces.Discrete(env.n_subtasks),
                cg=spaces.Discrete(2),
                cr=spaces.Discrete(2),
            ))
        self.last_g = None
        self.g_one_hots = [np.eye(d) for d in task_space.nvec[0]]

    def step(self, action):
        actions = SubtasksActions(
            *np.split(action, len(self.action_space.spaces)))
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

        def broadcast3d(x):
            x = np.array(x)
            return np.broadcast_to(
                x.reshape(-1, 1, 1),
                (x.size, h, w),
            )

        subtask123 = broadcast3d(env.subtask)
        task_spec = broadcast3d(env.task)
        next_subtask = np.full((1, h, w), env.next_subtask)

        obs_parts = [obs, subtask123, task_spec, next_subtask]
        stack = np.vstack(obs_parts)
        # print('obs', obs.shape)
        # print('interaction', interaction_one_hot.shape)
        # print('task_objects', task_objects_one_hot.shape)
        # print('task_spec', task_spec.shape)
        # print('iterate', iterate.shape)
        # print('stack', stack.shape)

        # names = ['obstacles'] + list(env.object_types) + ['ice', 'agent'] + \
        #         list(env.interactions) + ['task objects']
        # assert len(obs) == len(names)
        # for array, name in zip(obs, names):
        #     print(name)
        #     print(array)

        return stack.astype(float)

    def render(self, mode='human'):
        super().render(mode=mode)
        if self.last_g is not None:
            env = self.env.unwrapped
            g_type, g_count, g_obj = tuple(env.task[self.last_g])
            print(
                'Assigned subtask:',
                env.interactions[g_type],
                g_count,
                env.object_types[g_obj],
            )
        input('paused')


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
            dtype=self.observation_space.dtype,
        )

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
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = "cpu"
        # TODO: Fix data types

    @staticmethod
    def extract_numpy(obs):
        if isinstance(obs, dict):
            # print("VecPyTorch")
            # for k, x in obs.items():
            #     print(k, x.shape)
            return np.hstack([x.reshape(x.shape[0], -1) for x in obs.values()])
        if not isinstance(obs, (list, tuple)):
            return obs
        assert len(obs) == 1
        return obs[0]

    def reset(self):
        obs = self.extract_numpy(self.venv.reset())
        return torch.from_numpy(obs).float().to(self.device)

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

    def evaluate(self):
        self.venv.evaluate()

    def train(self):
        self.venv.train()


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs):
        if self.ob_rms:
            if self.training:
                self.ob_rms.update(obs)
            obs = np.clip(
                (obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon),
                -self.clipob,
                self.clipob,
            )
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

        self.stacked_obs = torch.zeros((venv.num_envs,) + low.shape)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype
        )
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, : -self.shape_dim0] = self.stacked_obs[:, self.shape_dim0 :]
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0 :] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        self.stacked_obs[:, -self.shape_dim0 :] = obs
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
                    obs.reshape(len(obs), -1).T, nvec.reshape(len(nvec), -1).T
                ):
                    yield onehot(o, n)

            return np.concatenate(list(one_hots()), axis=-1)


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, "venv"):
        return get_vec_normalize(venv.venv)

    return None
