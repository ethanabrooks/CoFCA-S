import inspect
import itertools
import os
import sys
from collections import namedtuple
from pathlib import Path
from pprint import pprint
from typing import Dict, Optional

import gym
import ray
import torch
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from tensorboardX import SummaryWriter

from aggregator import SumAcrossEpisode, InfosAggregator, EvalWrapper
from common.vec_env.dummy_vec_env import DummyVecEnv
from common.vec_env.subproc_vec_env import SubprocVecEnv
from common.vec_env.util import set_seeds
from networks import Agent, AgentOutputs, MLPBase
from ppo import PPO
from rollouts import RolloutStorage
from utils import get_device
from wrappers import VecPyTorch

EpochOutputs = namedtuple("EpochOutputs", "obs reward done infos act masks")


class Trainer:
    def run(self, **config):
        config = self.structure_config(**config)
        self.train(**config)

    def structure_config(self, **config):
        agent_args = {}
        rollouts_args = {}
        ppo_args = {}
        gen_args = {}
        for k, v in config.items():
            if k in inspect.signature(self.build_agent).parameters:
                agent_args[k] = v
            if k in inspect.signature(Agent.__init__).parameters:
                agent_args[k] = v
            if k in inspect.signature(MLPBase.__init__).parameters:
                agent_args[k] = v
            if k in inspect.signature(RolloutStorage.__init__).parameters:
                rollouts_args[k] = v
            if k in inspect.signature(PPO.__init__).parameters:
                ppo_args[k] = v
            if k in inspect.signature(self.train).parameters or k not in (
                list(agent_args.keys())
                + list(rollouts_args.keys())
                + list(ppo_args.keys())
            ):
                gen_args[k] = v
        config = dict(
            agent_args=agent_args,
            rollouts_args=rollouts_args,
            ppo_args=ppo_args,
            **gen_args,
        )
        return config

    @staticmethod
    def save_checkpoint(tmp_checkpoint_dir, ppo, agent, step):
        modules = dict(
            optimizer=ppo.optimizer, agent=agent
        )  # type: Dict[str, torch.nn.Module]
        # if isinstance(self.envs.venv, VecNormalize):
        #     modules.update(vec_normalize=self.envs.venv)
        state_dict = {name: module.state_dict() for name, module in modules.items()}
        save_path = Path(tmp_checkpoint_dir, f"checkpoint.pt")
        torch.save(dict(step=step, **state_dict), save_path)
        print(f"Saved parameters to {save_path}")

    @staticmethod
    def load_checkpoint(checkpoint_path, ppo, agent, device):
        state_dict = torch.load(checkpoint_path, map_location=device)
        agent.load_state_dict(state_dict["agent"])
        ppo.optimizer.load_state_dict(state_dict["optimizer"])
        # if isinstance(self.envs.venv, VecNormalize):
        #     self.envs.venv.load_state_dict(state_dict["vec_normalize"])
        print(f"Loaded parameters from {checkpoint_path}.")
        return state_dict.get("step", -1) + 1

    def train(
        self,
        agent_args: dict,
        cuda: bool,
        cuda_deterministic: bool,
        env_args: dict,
        env_id: str,
        log_dir: Optional[str],
        log_interval: int,
        name: str,
        normalize: float,
        num_iterations: int,
        num_processes: int,
        ppo_args: dict,
        render_eval: bool,
        rollouts_args: dict,
        seed: int,
        save_interval: int,
        synchronous: bool,
        train_steps: int,
        use_tune: bool,
        eval_interval: int = None,
        eval_steps: int = None,
        no_eval: bool = False,
        load_path: Path = None,
        render: bool = False,
    ):
        # Properly restrict pytorch to not consume extra resources.
        #  - https://github.com/pytorch/pytorch/issues/975
        #  - https://github.com/ray-project/ray/issues/3609
        torch.set_num_threads(1)
        os.environ["OMP_NUM_THREADS"] = "1"

        if use_tune:
            report_iterator = None
        else:

            def report_generator():
                writer = SummaryWriter(logdir=str(log_dir)) if log_dir else None

                for i in itertools.count():
                    if i % log_interval == 0:
                        values = yield
                        pprint(values)
                        if writer:
                            for k, v in values.items():
                                writer.add_scalar(k, v, i)

            report_iterator = report_generator()

        def make_vec_envs(evaluation):
            def env_thunk(rank):
                return lambda: self.make_env(
                    rank=rank, evaluation=evaluation, **env_args
                )

            env_fns = [env_thunk(i) for i in range(num_processes)]
            use_dummy = len(env_fns) == 1 or sys.platform == "darwin" or synchronous
            return VecPyTorch(
                DummyVecEnv(env_fns, render=render)
                if use_dummy
                else SubprocVecEnv(env_fns)
            )

        def run_epoch(obs, rnn_hxs, masks, envs, num_steps):
            for _ in range(num_steps):
                with torch.no_grad():
                    act = agent(
                        inputs=obs, rnn_hxs=rnn_hxs, masks=masks
                    )  # type: AgentOutputs

                action = envs.preprocess(act.action)
                # Observe reward and next obs
                obs, reward, done, infos = envs.step(action)

                # If done then clean the history of observations.
                masks = torch.tensor(
                    1 - done, dtype=torch.float32, device=obs.device
                ).unsqueeze(1)
                yield EpochOutputs(
                    obs=obs, reward=reward, done=done, infos=infos, act=act, masks=masks
                )

                rnn_hxs = act.rnn_hxs

        if render_eval and not render:
            eval_interval = 1
        if render or render_eval:
            ppo_args.update(ppo_epoch=0)
            num_processes = 1
            cuda = False
        cuda &= torch.cuda.is_available()

        # reproducibility
        set_seeds(cuda, cuda_deterministic, seed)

        if cuda:
            device = torch.device("cuda") if name else get_device(name)
        else:
            device = torch.device("cpu")
        print("Using device", device)

        train_envs = make_vec_envs(evaluation=False)
        try:
            train_envs.to(device)
            agent = self.build_agent(envs=train_envs, **agent_args)
            rollouts = RolloutStorage(
                num_steps=train_steps,
                obs_space=train_envs.observation_space,
                action_space=train_envs.action_space,
                recurrent_hidden_state_size=agent.recurrent_hidden_state_size,
                **rollouts_args,
            )

            # copy to device
            if cuda:
                agent.to(device)
                rollouts.to(device)

            ppo = PPO(agent=agent, **ppo_args)
            train_report = SumAcrossEpisode()
            train_infos = InfosAggregator()
            start = 0
            if load_path:
                start = self.load_checkpoint(load_path, ppo, agent, device)

            for i in range(start, num_iterations):
                eval_report = EvalWrapper(SumAcrossEpisode())
                eval_infos = EvalWrapper(InfosAggregator())
                if eval_interval and not no_eval and i % eval_interval == 0:
                    # vec_norm = get_vec_normalize(eval_envs)
                    # if vec_norm is not None:
                    #     vec_norm.eval()
                    #     vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

                    # self.envs.evaluate()
                    eval_masks = torch.zeros(num_processes, 1, device=device)
                    eval_envs = make_vec_envs(evaluation=True)
                    eval_envs.to(device)
                    with agent.recurrent_module.evaluating(eval_envs.observation_space):
                        eval_recurrent_hidden_states = torch.zeros(
                            num_processes,
                            agent.recurrent_hidden_state_size,
                            device=device,
                        )

                        for output in run_epoch(
                            obs=eval_envs.reset(),
                            rnn_hxs=eval_recurrent_hidden_states,
                            masks=eval_masks,
                            envs=eval_envs,
                            num_steps=eval_steps,
                        ):
                            eval_report.update(
                                reward=output.reward.cpu().numpy(),
                                dones=output.done,
                            )
                            eval_infos.update(*output.infos, dones=output.done)
                    eval_envs.close()

                rollouts.obs[0].copy_(train_envs.reset())

                for output in run_epoch(
                    obs=rollouts.obs[0],
                    rnn_hxs=rollouts.recurrent_hidden_states[0],
                    masks=rollouts.masks[0],
                    envs=train_envs,
                    num_steps=train_steps,
                ):
                    train_report.update(
                        reward=output.reward.cpu().numpy(),
                        dones=output.done,
                    )
                    train_infos.update(*output.infos, dones=output.done)
                    rollouts.insert(
                        obs=output.obs,
                        recurrent_hidden_states=output.act.rnn_hxs,
                        actions=output.act.action,
                        action_log_probs=output.act.action_log_probs,
                        values=output.act.value,
                        rewards=output.reward,
                        masks=output.masks,
                    )

                with torch.no_grad():
                    next_value = agent.get_value(
                        rollouts.obs[-1],
                        rollouts.recurrent_hidden_states[-1],
                        rollouts.masks[-1],
                    )

                rollouts.compute_returns(next_value.detach())
                train_results = ppo.update(rollouts)
                rollouts.after_update()

                total_num_steps = num_processes * train_steps * (i + 1)
                if total_num_steps % log_interval == 0:
                    report = dict(
                        **train_results,
                        **dict(train_report.items()),
                        **dict(eval_report.items()),
                    )
                    if use_tune:
                        tune.report(**report)
                    else:
                        assert report_iterator is not None
                        report_iterator.send(report)
                    train_report = SumAcrossEpisode()
                    train_infos = InfosAggregator()
                if save_interval and total_num_steps % save_interval == 0:
                    if use_tune:
                        with tune.checkpoint_dir(i) as _dir:
                            checkpoint_dir = _dir
                    else:
                        checkpoint_dir = Path(log_dir, str(i))

                    self.save_checkpoint(checkpoint_dir, ppo=ppo, agent=agent, step=i)

        finally:
            train_envs.close()

    @staticmethod
    def build_agent(envs, **agent_args):
        return Agent(envs.observation_space.shape, envs.action_space, **agent_args)

    @staticmethod
    def make_env(env_id, seed, rank, evaluation, **kwargs):
        env = gym.make(env_id, **kwargs)
        env.seed(seed + rank)
        return env

    @classmethod
    def main(
        cls,
        gpus_per_trial: float,
        cpus_per_trial: float,
        log_dir: str,
        num_samples: int,
        name: str,
        config: dict,
        **kwargs,
    ):
        for k, v in kwargs.items():
            if k not in config or v is not None:
                config[k] = v

        config.update(name=name, log_dir=log_dir)
        if num_samples is None:
            print("Not using tune, because num_samples was not specified")
            config.update(use_tune=False)
            cls().run(**config)
        else:
            local_mode = num_samples is None
            ray.init(dashboard_host="127.0.0.1", local_mode=local_mode)
            resources_per_trial = dict(gpu=gpus_per_trial, cpu=cpus_per_trial)
            config.update(use_tune=True)

            if local_mode:
                print("Using local mode because num_samples is None")
                kwargs = dict()
            else:
                kwargs = dict(
                    search_alg=HyperOptSearch(config, metric="eval_reward"),
                    num_samples=num_samples,
                )
            if log_dir is not None:
                kwargs.update(local_dir=log_dir)

            def run(c):
                cls().run(**c)

            tune.run(
                run,
                name=name,
                config=config,
                resources_per_trial=resources_per_trial,
                **kwargs,
            )
