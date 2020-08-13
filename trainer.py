import abc
import itertools
import sys
import time
from collections import defaultdict, deque, namedtuple
from pathlib import Path
from pprint import pprint
from typing import Dict

import gym
import numpy as np
import torch
from ray import tune
from tensorboardX import SummaryWriter

from agent import Agent, AgentOutputs
from common.vec_env.dummy_vec_env import DummyVecEnv
from common.vec_env.subproc_vec_env import SubprocVecEnv
from common.vec_env.util import set_seeds
from ppo import PPO
from rollouts import RolloutStorage
from utils import k_scalar_pairs
from wrappers import VecPyTorch

EpochOutputs = namedtuple("EpochOutputs", "obs reward done infos act masks")


class Trainer(abc.ABC):
    def __init__(
        self,
        agent_args: dict,
        cuda: bool,
        cuda_deterministic: bool,
        env_id: str,
        eval_interval: int,
        eval_steps: int,
        load_path: Path,
        log_interval: int,
        normalize: float,
        num_batch: int,
        num_epochs: int,
        num_processes: int,
        ppo_args: dict,
        render: bool,
        render_eval: bool,
        rollouts_args: dict,
        save_interval: int,
        seed: int,
        synchronous: bool,
        train_steps: int,
        use_tune=False,
        log_dir: Path = "dumb",
        no_eval=False,
    ):
        class EpochCounter:
            def __init__(self):
                self.episode_rewards = []
                self.episode_time_steps = []
                self.rewards = np.zeros(num_processes)
                self.time_steps = np.zeros(num_processes)

            def update(self, reward, done):
                self.rewards += reward.numpy()
                self.time_steps += np.ones_like(done)
                self.episode_rewards += list(self.rewards[done])
                self.episode_time_steps += list(self.time_steps[done])
                self.rewards[done] = 0
                self.time_steps[done] = 0

            def reset(self):
                self.episode_rewards = []
                self.episode_time_steps = []

        def make_vec_envs(evaluation):
            def env_thunk(rank):
                return self.make_env(
                    seed=seed, rank=rank, evaluation=evaluation, env_id=env_id
                )

            env_fns = [lambda: env_thunk(i) for i in range(num_processes)]
            use_dummy = len(env_fns) == 1 or sys.platform == "darwin" or synchronous
            return VecPyTorch(
                DummyVecEnv(env_fns, render=render)
                if use_dummy
                else SubprocVecEnv(env_fns)
            )

        def run_epoch(obs, rnn_hxs, masks, envs, num_steps):
            episode_counter = defaultdict(list)
            for _ in range(num_steps):
                with torch.no_grad():
                    act = self.agent(
                        inputs=obs, rnn_hxs=rnn_hxs, masks=masks
                    )  # type: AgentOutputs

                # Observe reward and next obs
                obs, reward, done, infos = envs.step(act.action)
                self.process_infos(episode_counter, done, infos, **act.log)

                # If done then clean the history of observations.
                masks = torch.tensor(
                    1 - done, dtype=torch.float32, device=obs.device
                ).unsqueeze(1)
                yield EpochOutputs(
                    obs=obs, reward=reward, done=done, infos=infos, act=act, masks=masks
                )

                rnn_hxs = act.rnn_hxs

        def report(**kwargs):
            if use_tune:
                tune.report(**kwargs)
            else:
                pprint(kwargs)

        if not torch.cuda.is_available():
            cuda = False
        set_seeds(cuda=cuda, cuda_deterministic=cuda_deterministic, seed=seed)

        if log_dir:
            self.writer = SummaryWriter(logdir=str(log_dir))
        else:
            self.writer = None

        if render_eval and not render:
            eval_interval = 1
        if render or render_eval:
            ppo_args.update(ppo_epoch=0)
            num_processes = 1
            cuda = False

        device = torch.device("cuda" if cuda else "cpu")

        train_envs = make_vec_envs(evaluation=False)

        train_envs.to(device)
        self.agent = agent = self.build_agent(envs=train_envs, **agent_args)
        rollouts = RolloutStorage(
            num_steps=train_steps,
            num_processes=num_processes,
            obs_space=train_envs.observation_space,
            action_space=train_envs.action_space,
            recurrent_hidden_state_size=agent.recurrent_hidden_state_size,
            **rollouts_args,
        )

        # copy to device
        if cuda:
            agent.to(device)
            rollouts.to(device)

        ppo = PPO(agent=agent, num_batch=num_batch, **ppo_args)

        start = 0
        if load_path:
            state_dict = torch.load(load_path, map_location=device)
            agent.load_state_dict(state_dict["agent"])
            ppo.optimizer.load_state_dict(state_dict["optimizer"])
            start = state_dict.get("step", -1) + 1
            # if isinstance(self.envs.venv, VecNormalize):
            #     self.envs.venv.load_state_dict(state_dict["vec_normalize"])
            print(f"Loaded parameters from {load_path}.")

        train_counter = EpochCounter()

        rollouts.obs[0].copy_(train_envs.reset())

        for i in range(start, num_epochs):
            for epoch_output in run_epoch(
                obs=rollouts.obs[0],
                rnn_hxs=rollouts.recurrent_hidden_states[0],
                masks=rollouts.masks[0],
                envs=train_envs,
                num_steps=train_steps,
            ):
                train_counter.update(reward=epoch_output.reward, done=epoch_output.done)
                rollouts.insert(
                    obs=epoch_output.obs,
                    recurrent_hidden_states=epoch_output.act.rnn_hxs,
                    actions=epoch_output.act.action,
                    action_log_probs=epoch_output.act.action_log_probs,
                    values=epoch_output.act.value,
                    rewards=epoch_output.reward,
                    masks=epoch_output.masks,
                )

            with torch.no_grad():
                next_value = agent.get_value(
                    rollouts.obs[-1],
                    rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1],
                ).detach()

            rollouts.compute_returns(next_value=next_value)
            train_results = ppo.update(rollouts)
            rollouts.after_update()

            eval_counter = dict()
            if eval_interval and not no_eval and i % eval_interval == 0:
                # vec_norm = get_vec_normalize(eval_envs)
                # if vec_norm is not None:
                #     vec_norm.eval()
                #     vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

                # self.envs.evaluate()
                eval_masks = torch.zeros(num_processes, 1, device=device)
                eval_envs = make_vec_envs(evaluation=True)
                eval_envs.to(device)
                with agent.network.evaluating(eval_envs.observation_space):
                    eval_recurrent_hidden_states = torch.zeros(
                        num_processes, agent.recurrent_hidden_state_size, device=device
                    )
                    eval_counter = EpochCounter()
                    for epoch_output in run_epoch(
                        obs=eval_envs.reset(),
                        rnn_hxs=eval_recurrent_hidden_states,
                        masks=eval_masks,
                        envs=eval_envs,
                        num_steps=eval_steps,
                    ):
                        eval_counter.update(
                            reward=epoch_output.reward, done=epoch_output.done
                        )

                eval_envs.close()
                eval_counter = dict(
                    eval_rewards=eval_counter.episode_rewards,
                    eval_time_steps=eval_counter.episode_time_steps,
                )

            if i % log_interval == 0:
                result = dict(
                    rewards=train_counter.episode_rewards,
                    time_steps=train_counter.episode_time_steps,
                    **eval_counter,
                    **train_results,
                )
                for k, v in k_scalar_pairs(**result):
                    if self.writer:
                        self.writer.add_scalar(k, v, i)
                train_counter.reset()

            if use_tune and save_interval and i % save_interval == 0:
                modules = dict(
                    optimizer=ppo.optimizer, agent=agent
                )  # type: Dict[str, torch.nn.Module]
                # if isinstance(self.envs.venv, VecNormalize):
                #     modules.update(vec_normalize=self.envs.venv)
                state_dict = {
                    name: module.state_dict() for name, module in modules.items()
                }
                with tune.checkpoint_dir(i) as checkpoint_dir:
                    save_path = Path(checkpoint_dir, f"checkpoint.pt")
                    torch.save(dict(step=i, **state_dict), save_path)
                    print(f"Saved parameters to {save_path}")

    @staticmethod
    def process_infos(episode_counter, done, infos, **act_log):
        for d in infos:
            for k, v in d.items():
                episode_counter[k] += v if type(v) is list else [float(v)]
        for k, v in act_log.items():
            episode_counter[k] += v if type(v) is list else [float(v)]

    @staticmethod
    def build_agent(envs, **agent_args):
        return Agent(envs.observation_space.shape, envs.action_space, **agent_args)

    @staticmethod
    def make_env(env_id, seed, rank, evaluation):
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
