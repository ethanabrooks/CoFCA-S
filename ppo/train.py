import functools
import itertools
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict

import gym
import numpy as np
import torch
from gym.wrappers import TimeLimit
from tqdm import tqdm

from common.atari_wrappers import wrap_deepmind
from common.vec_env.dummy_vec_env import DummyVecEnv
from common.vec_env.subproc_vec_env import SubprocVecEnv
from ppo.agent import Agent, AgentValues
from ppo.storage import RolloutStorage
from ppo.update import PPO
from ppo.utils import k_scalar_pairs
from ppo.wrappers import AddTimestep, TransposeImage, VecPyTorch, VecPyTorchFrameStack

try:
    import dm_control2gym
except ImportError:
    pass


# noinspection PyAttributeOutsideInit
class Train:
    def setup(
        self,
        num_steps,
        num_processes,
        seed,
        cuda_deterministic,
        cuda,
        time_limit,
        gamma,
        normalize,
        log_interval,
        eval_interval,
        use_gae,
        tau,
        ppo_args,
        agent_args,
        render,
        render_eval,
        load_path,
        success_reward,
        synchronous,
        batch_size,
        env_args,
    ):
        if render_eval and not render:
            eval_interval = 1
        if render:
            ppo_args.update(ppo_epoch=0)
            num_processes = 1
            cuda = False
        self.success_reward = success_reward

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        cuda &= torch.cuda.is_available()
        if cuda and cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        self.device = "cpu"
        if cuda:
            self.device = self.get_device()
        # print("Using device", self.device)

        torch.set_num_threads(1)

        self.make_eval_envs = lambda: self.make_vec_envs(
            **env_args,
            # env_id=env_id,
            time_limit=time_limit,
            num_processes=num_processes,
            # add_timestep=add_timestep,
            render=render_eval,
            seed=seed + num_processes,
            gamma=gamma if normalize else None,
            evaluation=True,
            synchronous=True if render_eval else synchronous,
        )
        self.make_train_envs = lambda: self.make_vec_envs(
            **env_args,
            seed=seed,
            gamma=(gamma if normalize else None),
            render=render,
            synchronous=True if render else synchronous,
            evaluation=False,
            num_processes=num_processes,
            time_limit=time_limit,
        )
        envs = self.make_train_envs()
        self.agent = self.build_agent(envs=envs, **agent_args)
        self.num_steps = num_steps
        self.processes = num_processes
        self.rollouts = RolloutStorage(
            num_steps=self.num_steps,
            num_processes=self.processes,
            obs_space=envs.observation_space,
            action_space=envs.action_space,
            recurrent_hidden_state_size=self.agent.recurrent_hidden_state_size,
            use_gae=use_gae,
            gamma=gamma,
            tau=tau,
        )
        if cuda:
            tick = time.time()
            self.agent.to(self.device)
            self.rollouts.to(self.device)
            print("Values copied to GPU in", time.time() - tick, "seconds")

        self.ppo = PPO(agent=self.agent, batch_size=batch_size, **ppo_args)
        counter = Counter()

        if load_path:
            self._restore(load_path)

        self.interval = log_interval
        self.eval_interval = eval_interval
        self.counter = counter
        self.time_limit = time_limit
        self.i = 0
        envs.close()
        self.tick = time.time()
        self.log_progress = None
        del envs

    def run(self):
        for _ in itertools.count():
            self._train()

    def _train(self):
        envs = self.make_train_envs()
        envs.to(self.device)
        obs = envs.reset()
        self.rollouts.obs[0].copy_(obs)
        for _ in tqdm(range(self.eval_interval), desc="eval"):
            if self.i % self.interval == 0:
                self.log_progress = tqdm(total=self.interval, desc="log ")
            self.i += 1
            epoch_counter = self.run_epoch(
                obs=self.rollouts.obs[0],
                rnn_hxs=self.rollouts.recurrent_hidden_states[0],
                masks=self.rollouts.masks[0],
                envs=envs,
                num_steps=self.num_steps,
                counter=self.counter,
            )

            with torch.no_grad():
                next_value = self.agent.get_value(
                    self.rollouts.obs[-1],
                    self.rollouts.recurrent_hidden_states[-1],
                    self.rollouts.masks[-1],
                ).detach()

            self.rollouts.compute_returns(next_value=next_value)
            train_results = self.ppo.update(self.rollouts)
            self.rollouts.after_update()
            total_num_steps = (self.i + 1) * self.processes * self.num_steps
            self.log_progress.update()
            if self.i % self.interval == 0:
                start = self.tick
                self.tick = time.time()
                fps = total_num_steps / (self.tick - start)
                self._log_result(
                    dict(k_scalar_pairs(fps=fps, **epoch_counter, **train_results))
                )

        envs.close()
        del envs
        envs = self.make_eval_envs()
        envs.to(self.device)
        # vec_norm = get_vec_normalize(eval_envs)
        # if vec_norm is not None:
        #     vec_norm.eval()
        #     vec_norm.ob_rms = get_vec_normalize(envs).ob_rms
        eval_recurrent_hidden_states = torch.zeros(
            self.processes, self.agent.recurrent_hidden_state_size, device=self.device
        )
        eval_masks = torch.zeros(self.processes, 1, device=self.device)
        eval_counter = Counter()
        print("Evaluating....")
        eval_values = self.run_epoch(
            envs=envs,
            obs=envs.reset(),
            rnn_hxs=eval_recurrent_hidden_states,
            masks=eval_masks,
            num_steps=max(self.num_steps, self.time_limit)
            if self.time_limit
            else self.num_steps,
            counter=eval_counter,
        )
        envs.close()
        del envs
        return dict(k_scalar_pairs(**{f"eval_{k}": v for k, v in eval_values.items()}))

    def run_epoch(self, obs, rnn_hxs, masks, envs, num_steps, counter):
        # noinspection PyTypeChecker
        episode_counter = Counter(rewards=[], time_steps=[], success=[])
        for step in range(num_steps):
            with torch.no_grad():
                act = self.agent(
                    inputs=obs, rnn_hxs=rnn_hxs, masks=masks
                )  # type: AgentValues

            # Observe reward and next obs
            obs, reward, done, infos = envs.step(act.action)

            for d in infos:
                for k, v in d.items():
                    episode_counter.update({k: float(v) / num_steps / len(infos)})

            # track rewards
            counter["reward"] += reward.numpy()
            counter["time_step"] += np.ones_like(done)
            episode_rewards = counter["reward"][done]
            episode_counter["rewards"] += list(episode_rewards)
            if self.success_reward is not None:
                episode_counter["success"] += list(
                    episode_rewards >= self.success_reward
                )
                # if np.any(episode_rewards < self.success_reward):
                #     import ipdb
                #
                #     ipdb.set_trace()

            episode_counter["time_steps"] += list(counter["time_step"][done])
            counter["reward"][done] = 0
            counter["time_step"][done] = 0

            # If done then clean the history of observations.
            masks = torch.tensor(
                1 - done, dtype=torch.float32, device=obs.device
            ).unsqueeze(1)
            rnn_hxs = act.rnn_hxs
            if self.rollouts is not None:
                self.rollouts.insert(
                    obs=obs,
                    recurrent_hidden_states=act.rnn_hxs,
                    actions=act.action,
                    action_log_probs=act.action_log_probs,
                    values=act.value,
                    rewards=reward,
                    masks=masks,
                )

        return episode_counter

    @staticmethod
    def build_agent(envs, **agent_args):
        return Agent(envs.observation_space.shape, envs.action_space, **agent_args)

    @staticmethod
    def make_env(env_id, seed, rank, add_timestep, time_limit, evaluation):
        if env_id.startswith("dm"):
            _, domain, task = env_id.split(".")
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:
            env = gym.make(env_id)

        is_atari = hasattr(gym.envs, "atari") and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv
        )

        env.seed(seed + rank)

        obs_shape = env.observation_space.shape

        if add_timestep and len(obs_shape) == 1 and str(env).find("TimeLimit") > -1:
            env = AddTimestep(env)

        if is_atari and len(env.observation_space.shape) == 3:
            env = wrap_deepmind(env)

        # elif len(env.observation_space.shape) == 3:
        #     raise NotImplementedError(
        #         "CNN models work only for atari,\n"
        #         "please use a custom wrapper for a custom pixel input env.\n"
        #         "See wrap_deepmind for an example.")

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env)

        if time_limit is not None:
            env = TimeLimit(env, max_episode_steps=time_limit)

        return env

    def make_vec_envs(
        self, num_processes, gamma, render, synchronous, num_frame_stack=None, **kwargs
    ):
        envs = [
            functools.partial(self.make_env, rank=i, **kwargs)
            for i in range(num_processes)
        ]

        if len(envs) == 1 or sys.platform == "darwin" or synchronous:
            envs = DummyVecEnv(envs, render=render)
        else:
            envs = SubprocVecEnv(envs)

        # if (
        # envs.observation_space.shape
        # and len(envs.observation_space.shape) == 1
        # ):
        # if gamma is None:
        # envs = VecNormalize(envs, ret=False)
        # else:
        # envs = VecNormalize(envs, gamma=gamma)

        envs = VecPyTorch(envs)

        if num_frame_stack is not None:
            envs = VecPyTorchFrameStack(envs, num_frame_stack)
        # elif len(envs.observation_space.shape) == 3:
        #     envs = VecPyTorchFrameStack(envs, 4, device)

        return envs

    def _save(self, checkpoint_dir):
        modules = dict(
            optimizer=self.ppo.optimizer, agent=self.agent
        )  # type: Dict[str, torch.nn.Module]
        # if isinstance(self.envs.venv, VecNormalize):
        #     modules.update(vec_normalize=self.envs.venv)
        state_dict = {name: module.state_dict() for name, module in modules.items()}
        save_path = Path(checkpoint_dir, "checkpoint.pt")
        torch.save(dict(step=self.i, **state_dict), save_path)
        print(f"Saved parameters to {save_path}")
        return save_path

    def _restore(self, checkpoint):
        load_path = checkpoint
        state_dict = torch.load(load_path, map_location=self.device)
        self.agent.load_state_dict(state_dict["agent"])
        self.ppo.optimizer.load_state_dict(state_dict["optimizer"])
        self.i = state_dict.get("step", -1) + 1
        # if isinstance(self.envs.venv, VecNormalize):
        #     self.envs.venv.load_state_dict(state_dict["vec_normalize"])
        print(f"Loaded parameters from {load_path}.")

    def _log_result(self, result):
        pass
