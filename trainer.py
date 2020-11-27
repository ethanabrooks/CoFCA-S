import inspect
import itertools
import os
import time
from argparse import ArgumentParser
from collections import namedtuple, Counter, defaultdict
from pathlib import Path
from pprint import pprint
from typing import Dict, DefaultDict, Union

import gym
import torch
import torch.nn as nn
import wandb

import arguments
from agents import Agent, AgentOutputs, MLPBase
from aggregator import EpisodeAggregator, InfosAggregator, EvalWrapper
from common.vec_env.dummy_vec_env import DummyVecEnv
from common.vec_env.subproc_vec_env import SubprocVecEnv
from common.vec_env.util import set_seeds
from ppo import PPO
from rollouts import RolloutStorage
from wrappers import VecPyTorch

EpochOutputs = namedtuple("EpochOutputs", "obs reward done infos act masks")
CHECKPOINT_NAME = "checkpoint.pt"


class Trainer:
    metric = "reward"

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
        if config["render"]:
            config["num_processes"] = 1

        def parameters(*ms):
            for method in ms:
                yield from inspect.signature(method).parameters

        args = defaultdict(dict)
        args_to_methods = cls.args_to_methods()
        for k, v in config.items():
            if k in ("_wandb", "wandb_version"):
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
    def save_checkpoint(save_path: Path, ppo: PPO, agent: Agent, step: int):
        modules = dict(
            optimizer=ppo.optimizer, agent=agent
        )  # type: Dict[str, torch.nn.Module]
        # if isinstance(self.envs.venv, VecNormalize):
        #     modules.update(vec_normalize=self.envs.venv)
        state_dict = {name: module.state_dict() for name, module in modules.items()}
        torch.save(dict(step=step, **state_dict), save_path)
        print(f"Saved parameters to {save_path}")

    @staticmethod
    def load_checkpoint(checkpoint_path, ppo, agent, device):
        state_dict = torch.load(str(checkpoint_path), map_location=device)
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
        log_dir: Path,
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
        save_path = Path(log_dir, CHECKPOINT_NAME)

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
                self.load_checkpoint(load_path, ppo, agent, device)

            print("resetting environment...")
            rollouts.obs[0].copy_(train_envs.reset())
            print("Reset environment")
            frames_per_update = train_steps * num_processes
            frames = Counter()
            time_spent = Counter()
            iter_tick = None

            for i in itertools.count():
                frames.update(so_far=frames_per_update)
                done = frames["so_far"] >= num_frames
                if not no_eval and (
                    i == 0
                    or done
                    or (eval_interval and frames["since_eval"] > eval_interval)
                ):
                    print("Evaluating...")
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
                            frames=frames["so_far"],
                            log_dir=log_dir,
                        )
                        print("Done evaluating...")
                    eval_envs.close()
                    rollouts.obs[0].copy_(train_envs.reset())
                    rollouts.masks[0] = 1
                    rollouts.recurrent_hidden_states[0] = 0
                if done or i == 0 or frames["since_log"] > log_interval:
                    log_tick = time.time()
                    frames["since_log"] = 0
                    report = dict(
                        **train_results,
                        **dict(train_report.items()),
                        **dict(train_infos.items()),
                        time_logging=time_spent["logging"],
                        time_saving=time_spent["saving"],
                        frames=frames["so_far"],
                        log_dir=log_dir,
                    )
                    if iter_tick is not None:
                        report.update(time_this_iter=time.time() - iter_tick)
                    iter_tick = time.time()
                    self.report(**report)
                    train_report.reset()
                    train_infos.reset()
                    time_spent["logging"] += time.time() - log_tick

                if done or (save_interval and frames["since_save"] > save_interval):
                    tick = time.time()
                    frames["since_save"] = 0
                    self.save_checkpoint(
                        save_path,
                        ppo=ppo,
                        agent=agent,
                        step=i,
                    )
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
    def report(frames: int, log_dir: Path, **kwargs):
        print("Frames:", frames)
        pprint(kwargs)
        try:
            wandb.log(kwargs, step=frames)
        except wandb.Error:
            pass

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
    def add_arguments(cls, parser):
        return arguments.add_arguments(parser)

    @classmethod
    def main(cls):
        parser = ArgumentParser()
        cls.add_arguments(parser)

        def run(config, no_wandb, **kwargs):
            if no_wandb:
                log_dir = Path("/tmp")
            else:
                wandb.init()
                kwargs.update(wandb.config.as_dict())
                log_dir = Path(wandb.run.dir)
            kwargs.update(config, log_dir=log_dir)
            cls().run(**cls.structure_config(**kwargs))

        run(**vars(parser.parse_args()))


if __name__ == "__main__":
    Trainer.main()
