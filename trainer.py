import inspect
import itertools
import os
import time
from argparse import ArgumentParser
from collections import namedtuple, Counter, defaultdict
from pathlib import Path
from pprint import pprint
from typing import Dict, Optional, DefaultDict, Union, List

import gym
import ray
import torch
import torch.nn as nn
import wandb
from wandb.sweeps.config import tune
from wandb.sweeps.config.tune.suggest.hyperopt import HyperOptSearch

from ray.tune.integration.wandb import wandb_mixin

import arguments
from agents import Agent, AgentOutputs, MLPBase
from aggregator import EpisodeAggregator, InfosAggregator, EvalWrapper
from common.vec_env.dummy_vec_env import DummyVecEnv
from common.vec_env.subproc_vec_env import SubprocVecEnv
from common.vec_env.util import set_seeds
from configs import default
from ppo import PPO
from rollouts import RolloutStorage
from wrappers import VecPyTorch

EpochOutputs = namedtuple("EpochOutputs", "obs reward done infos act masks")

CHECKPOINT_NAME = "checkpoint.pt"


class Trainer:
    metric = "reward"
    mode = "max"
    default = default

    @classmethod
    def args_to_methods(cls):
        return dict(
            agent_args=[
                cls.build_agent,
                Agent.__init__,
                MLPBase.__init__,
            ],
            rollouts_args=[RolloutStorage.__init__],
            ppo_args=[PPO.__init__],
            env_args=[cls.make_env],
            run_args=[cls.run],
        )

    @classmethod
    def structure_config(
        cls, **config
    ) -> DefaultDict[str, Dict[str, Union[bool, int, float]]]:
        def parameters(*ms):
            for method in ms:
                yield from inspect.signature(method).parameters

        args = defaultdict(dict)
        args_to_methods = cls.args_to_methods()
        for k, v in config.items():
            if k == "wandb":
                continue
            assigned = False
            for arg_name, methods in args_to_methods.items():
                if k in parameters(*methods):
                    args[arg_name][k] = v
                    assigned = True
            assert assigned, k
        run_args = args.pop("run_args")
        args.update(**run_args)
        return args

    @staticmethod
    def save_checkpoint(ppo, agent, step):
        modules = dict(
            optimizer=ppo.optimizer, agent=agent
        )  # type: Dict[str, torch.nn.Module]
        # if isinstance(self.envs.venv, VecNormalize):
        #     modules.update(vec_normalize=self.envs.venv)
        state_dict = {name: module.state_dict() for name, module in modules.items()}
        save_path = Path(wandb.run.dir, CHECKPOINT_NAME)
        torch.save(dict(step=step, **state_dict), save_path)
        print(f"Saved parameters to {save_path}")

    @staticmethod
    def load_checkpoint(ppo, agent, device):
        checkpoint_path = Path(wandb.run.dir, CHECKPOINT_NAME)
        state_dict = torch.load(checkpoint_path, map_location=device)
        agent.load_state_dict(state_dict["agent"])
        ppo.optimizer.load_state_dict(state_dict["optimizer"])
        # if isinstance(self.envs.venv, VecNormalize):
        #     self.envs.venv.load_state_dict(state_dict["vec_normalize"])
        print(f"Loaded parameters from {checkpoint_path}.")
        return state_dict.get("step", -1) + 1

    def run(
        self,
        agent_args: dict,
        cuda: bool,
        cuda_deterministic: bool,
        env_args: dict,
        log_interval: int,
        normalize: float,
        num_frames: int,
        num_processes: int,
        ppo_args: dict,
        rollouts_args: dict,
        seed: int,
        save_interval: int,
        synchronous: bool,
        train_steps: int,
        eval_interval: int = None,
        eval_steps: int = None,
        load_path: Path = None,
        no_eval: bool = False,
        render: bool = False,
        render_eval: bool = False,
    ):
        # Properly restrict pytorch to not consume extra resources.
        #  - https://github.com/pytorch/pytorch/issues/975
        #  - https://github.com/ray-project/ray/issues/3609
        torch.set_num_threads(1)
        os.environ["OMP_NUM_THREADS"] = "1"
        wandb.save(Path(wandb.run.dir, CHECKPOINT_NAME))

        def make_vec_envs(evaluating):
            def env_thunk(rank):
                return lambda: self.make_env(
                    rank=rank, evaluating=evaluating, **env_args
                )

            env_fns = [env_thunk(i) for i in range(num_processes)]
            return VecPyTorch(
                DummyVecEnv(env_fns, render=render)
                if len(env_fns) == 1 or synchronous
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
        set_seeds(seed)
        if cuda and cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        if cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print("Using device", device)

        train_envs = make_vec_envs(evaluating=False)
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
            train_report = EpisodeAggregator()
            train_infos = self.build_infos_aggregator()
            train_results = {}
            if load_path:
                wandb.restore(CHECKPOINT_NAME)
                self.load_checkpoint(ppo, agent, device)

            rollouts.obs[0].copy_(train_envs.reset())
            frames_per_update = train_steps * num_processes
            frames = Counter()
            time_spent = Counter()

            for i in itertools.count():
                frames.update(so_far=frames_per_update)
                done = frames["so_far"] >= num_frames
                if not no_eval and (
                    i == 0
                    or done
                    or (eval_interval and frames["since_eval"] > eval_interval)
                ):
                    eval_report = EvalWrapper(EpisodeAggregator())
                    eval_infos = EvalWrapper(InfosAggregator())
                    frames["since_eval"] = 0
                    # vec_norm = get_vec_normalize(eval_envs)
                    # if vec_norm is not None:
                    #     vec_norm.eval()
                    #     vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

                    # self.envs.evaluate()
                    eval_masks = torch.zeros(num_processes, 1, device=device)
                    eval_envs = make_vec_envs(evaluating=True)
                    eval_envs.to(device)
                    with agent.recurrent_module.evaluating(eval_envs.observation_space):
                        eval_recurrent_hidden_states = torch.zeros(
                            num_processes,
                            agent.recurrent_hidden_state_size,
                            device=device,
                        )
                        print("evaluating...")

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
                        self.report(
                            **dict(eval_report.items()),
                            **dict(eval_infos.items()),
                            training_iteration=frames["so_far"],
                        )
                        print("done evaluating")
                    eval_envs.close()
                    rollouts.obs[0].copy_(train_envs.reset())
                    rollouts.masks[0] = 1
                    rollouts.recurrent_hidden_states[0] = 0
                if done or frames["since_log"] > log_interval:
                    tick = time.time()
                    frames["since_log"] = 0
                    self.report(
                        **train_results,
                        **dict(train_report.items()),
                        **dict(train_infos.items()),
                        time_logging=time_spent["logging"],
                        time_saving=time_spent["saving"],
                        training_iteration=frames["so_far"],
                    )
                    train_report.reset()
                    train_infos.reset()
                    time_spent["logging"] += time.time() - tick

                if done or (save_interval and frames["since_save"] > save_interval):
                    tick = time.time()
                    frames["since_save"] = 0
                    self.save_checkpoint(ppo=ppo, agent=agent, step=i)
                    time_spent["saving"] += time.time() - tick

                if done:
                    break

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
                    frames.update(
                        since_save=num_processes,
                        since_log=num_processes,
                        since_eval=num_processes,
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

        finally:
            train_envs.close()

    @staticmethod
    def report(training_iteration: int, **kwargs):
        if tune.is_session_enabled():
            tune.report(**kwargs, training_iteration=training_iteration)
        else:
            pprint(kwargs)
        wandb.log(kwargs, global_step=training_iteration)

    def build_infos_aggregator(self):
        return InfosAggregator()

    @staticmethod
    def build_agent(envs, activation=nn.ReLU(), **agent_args):
        return Agent(
            envs.observation_space.shape,
            envs.action_space,
            activation=activation,
            **agent_args,
        )

    @staticmethod
    def make_env(env_id, seed, rank, evaluating, **kwargs):
        env = gym.make(env_id, **kwargs)
        env.seed(seed + rank)
        return env

    @classmethod
    def launch(
        cls,
        config: dict,
        cpus_per_trial: float,
        gpus_per_trial: float,
        group: str,
        notes: str,
        num_samples: int,
        project: str,
        seed: Optional[int],
        tags: List[str],
        wandb_key_file: Path,
        **kwargs,
    ):
        if config is None:
            config = cls.default
        if seed is not None:
            config.update(seed=seed)

        for k, v in kwargs.items():
            if k not in config or v is not None:
                if isinstance(v, bool):  # handle store_{true, false} differently
                    if v != cls.default[k]:  # flag used
                        config[k] = v
                else:
                    config[k] = v

        config.update(
            wandb=dict(
                project=project,
                api_key_file=str(wandb_key_file),
                group=group,
                notes=notes,
                save_code=True,
                tags=tags,
            ),
        )

        @wandb_mixin
        def run(c):
            c = cls.structure_config(**c)
            cls().run(**c)

        wandb.init(project=project)
        if num_samples is None:
            print("Not using tune, because num_samples was not specified")
            run(config)
        else:
            ray.init(dashboard_host="127.0.0.1")
            resources_per_trial = dict(
                gpu=gpus_per_trial if torch.cuda.is_available() else 0,
                cpu=cpus_per_trial,
            )

            kwargs = dict(
                search_alg=HyperOptSearch(
                    config, metric=cls.metric, mode=cls.mode, random_state_seed=seed
                ),
                num_samples=num_samples,
            )

            tune_config = tune.run(
                run,
                name=group,
                config=config,
                resources_per_trial=resources_per_trial,
                # loggers=DEFAULT_LOGGERS + (WandbLogger,),
                **kwargs,
            )

            # Save sweep as yaml config file
            tune_config.save("sweep-hyperopt.yaml")

            # Create the sweep
            wandb.sweep(tune_config)

    @classmethod
    def add_arguments(cls, parser):
        return arguments.add_arguments(parser)

    @classmethod
    def main(cls):
        parser = ArgumentParser()
        cls.add_arguments(parser)
        cls.launch(**vars(parser.parse_args()))


if __name__ == "__main__":
    Trainer.main()
