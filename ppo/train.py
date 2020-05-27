import abc
import functools
import itertools
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict

import dataset
import gym
import numpy as np
import torch
from gym.wrappers import TimeLimit
from tensorboardX import SummaryWriter
from tqdm import tqdm

from common.atari_wrappers import wrap_deepmind
from common.vec_env.dummy_vec_env import DummyVecEnv
from common.vec_env.subproc_vec_env import SubprocVecEnv
from ppo.agent import Agent, AgentValues
from ppo.control_flow.hdfstore import HDF5Store
from ppo.storage import RolloutStorage
from ppo.update import PPO
from ppo.utils import k_scalar_pairs, get_n_gpu, get_random_gpu
from ppo.wrappers import AddTimestep, TransposeImage, VecPyTorch, VecPyTorchFrameStack


# noinspection PyAttributeOutsideInit
class TrainBase(abc.ABC):
    def setup(
        self,
        num_steps,
        eval_steps,
        num_processes,
        seed,
        cuda_deterministic,
        cuda,
        time_limit,
        gamma,
        normalize,
        log_interval,
        eval_interval,
        no_eval,
        use_gae,
        tau,
        ppo_args,
        agent_args,
        render,
        render_eval,
        load_path,
        synchronous,
        num_batch,
        env_args,
        success_reward,
        use_tqdm,
    ):
        # Properly restrict pytorch to not consume extra resources.
        #  - https://github.com/pytorch/pytorch/issues/975
        #  - https://github.com/ray-project/ray/issues/3609
        torch.set_num_threads(1)
        os.environ["OMP_NUM_THREADS"] = "1"

        if render_eval and not render:
            eval_interval = 1
        if render or render_eval:
            ppo_args.update(ppo_epoch=0)
            num_processes = 1
            cuda = False

        # reproducibility
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        cuda &= torch.cuda.is_available()
        if cuda and cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        torch.set_num_threads(1)

        self.device = "cpu"
        if cuda:
            self.device = self.get_device()
        # print("Using device", self.device)

        self.envs = self.make_vec_envs(
            **env_args,
            seed=seed,
            gamma=(gamma if normalize else None),
            render=render,
            synchronous=True if render else synchronous,
            evaluation=False,
            num_processes=num_processes,
            time_limit=time_limit,
        )
        self.make_eval_envs = functools.partial(
            self.make_vec_envs,
            **env_args,
            seed=seed,
            gamma=(gamma if normalize else None),
            render=render,
            synchronous=True if render else synchronous,
            evaluation=True,
            num_processes=num_processes,
            time_limit=time_limit,
        )

        self.envs.to(self.device)
        self.agent = self.build_agent(envs=self.envs, **agent_args)
        self.rollouts = RolloutStorage(
            num_steps=num_steps,
            num_processes=num_processes,
            obs_space=self.envs.observation_space,
            action_space=self.envs.action_space,
            recurrent_hidden_state_size=self.agent.recurrent_hidden_state_size,
            use_gae=use_gae,
            gamma=gamma,
            tau=tau,
        )

        # copy to device
        if cuda:
            tick = time.time()
            self.agent.to(self.device)
            self.rollouts.to(self.device)
            print("Values copied to GPU in", time.time() - tick, "seconds")

        self.ppo = PPO(agent=self.agent, num_batch=num_batch, **ppo_args)
        self.counter = Counter()

        self.i = 0
        if load_path:
            self._restore(load_path)

        self.make_train_iterator = lambda: self.train_generator(
            num_steps=num_steps,
            num_processes=num_processes,
            eval_steps=eval_steps,
            log_interval=log_interval,
            eval_interval=eval_interval,
            no_eval=no_eval,
            use_tqdm=use_tqdm,
            success_reward=success_reward,
        )
        self.train_iterator = self.make_train_iterator()

    def _train(self):
        try:
            return next(self.train_iterator)
        except StopIteration:
            self.train_iterator = self.make_train_iterator()
            return self._train()

    def train_generator(
        self,
        num_steps,
        num_processes,
        eval_steps,
        log_interval,
        eval_interval,
        no_eval,
        success_reward,
        use_tqdm,
    ):
        if eval_interval and not no_eval:
            # vec_norm = get_vec_normalize(eval_envs)
            # if vec_norm is not None:
            #     vec_norm.eval()
            #     vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

            # self.envs.evaluate()
            eval_masks = torch.zeros(num_processes, 1, device=self.device)
            eval_counter = Counter()
            envs = self.make_eval_envs()
            envs.to(self.device)
            with self.agent.recurrent_module.evaluating(envs.observation_space):
                eval_recurrent_hidden_states = torch.zeros(
                    num_processes,
                    self.agent.recurrent_hidden_state_size,
                    device=self.device,
                )

                eval_result = self.run_epoch(
                    obs=envs.reset(),
                    rnn_hxs=eval_recurrent_hidden_states,
                    masks=eval_masks,
                    num_steps=eval_steps,
                    # max(num_steps, time_limit) if time_limit else num_steps,
                    counter=eval_counter,
                    success_reward=success_reward,
                    use_tqdm=use_tqdm,
                    rollouts=None,
                    envs=envs,
                )
            envs.close()
            eval_result = {f"eval_{k}": v for k, v in eval_result.items()}
        else:
            eval_result = {}
        # self.envs.train()
        obs = self.envs.reset()
        self.rollouts.obs[0].copy_(obs)
        tick = time.time()
        log_progress = None

        if eval_interval:
            eval_iterator = range(self.i % eval_interval, eval_interval)
            if use_tqdm:
                eval_iterator = tqdm(eval_iterator, desc="next eval")
        else:
            eval_iterator = itertools.count(self.i)

        for _ in eval_iterator:
            if self.i % log_interval == 0 and use_tqdm:
                log_progress = tqdm(total=log_interval, desc="next log")
            self.i += 1
            epoch_counter = self.run_epoch(
                obs=self.rollouts.obs[0],
                rnn_hxs=self.rollouts.recurrent_hidden_states[0],
                masks=self.rollouts.masks[0],
                num_steps=num_steps,
                counter=self.counter,
                success_reward=success_reward,
                use_tqdm=False,
                rollouts=self.rollouts,
                envs=self.envs,
            )

            with torch.no_grad():
                next_value = self.agent.get_value(
                    self.rollouts.obs[-1],
                    self.rollouts.recurrent_hidden_states[-1],
                    self.rollouts.masks[-1],
                )

            for vp, v in zip(
                [
                    self.rollouts.value_preds,
                    self.rollouts.value_preds2,
                    self.rollouts.value_preds3,
                ],
                [x.detach() for x in next_value],
            ):
                self.rollouts.compute_returns(vp, v)
            train_results = self.ppo.update(self.rollouts)
            self.rollouts.after_update()
            if log_progress is not None:
                log_progress.update()
            if self.i % log_interval == 0:
                total_num_steps = log_interval * num_processes * num_steps
                fps = total_num_steps / (time.time() - tick)
                tick = time.time()
                yield dict(
                    tick=tick, fps=fps, **epoch_counter, **train_results, **eval_result
                )

    def run_epoch(
        self,
        obs,
        rnn_hxs,
        masks,
        num_steps,
        counter,
        success_reward,
        use_tqdm,
        rollouts,
        envs,
    ):
        # noinspection PyTypeChecker
        episode_counter = defaultdict(list)
        iterator = range(num_steps)
        if use_tqdm:
            iterator = tqdm(iterator, desc="evaluating")
        for _ in iterator:
            with torch.no_grad():
                act = self.agent(
                    inputs=obs, rnn_hxs=rnn_hxs, masks=masks
                )  # type: AgentValues

            # Observe reward and next obs
            obs, reward, done, infos = envs.step(act.action)
            self.process_infos(episode_counter, done, infos, **act.log)

            # track rewards
            counter["reward"] += reward.numpy()
            counter["time_step"] += np.ones_like(done)
            episode_rewards = counter["reward"][done]
            episode_counter["rewards"] += list(episode_rewards)
            if success_reward is not None:
                # noinspection PyTypeChecker
                episode_counter["success"] += list(episode_rewards >= success_reward)

            episode_counter["time_steps"] += list(counter["time_step"][done])
            counter["reward"][done] = 0
            counter["time_step"][done] = 0

            # If done then clean the history of observations.
            masks = torch.tensor(
                1 - done, dtype=torch.float32, device=obs.device
            ).unsqueeze(1)
            rnn_hxs = act.rnn_hxs
            if rollouts is not None:
                rollouts.insert(
                    obs=obs,
                    recurrent_hidden_states=act.rnn_hxs,
                    actions=act.action,
                    action_log_probs=act.action_log_probs,
                    action_log_probs2=act.action_log_probs2,
                    action_log_probs3=act.action_log_probs3,
                    values=act.value,
                    values2=act.value2,
                    values3=act.value3,
                    rewards=reward,
                    masks=masks,
                )

        return dict(episode_counter)

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
    def make_env(env_id, seed, rank, add_timestep, time_limit, evaluation):
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
        self,
        num_processes,
        gamma,
        render,
        synchronous,
        env_id,
        add_timestep,
        seed,
        evaluation,
        time_limit,
        num_frame_stack=None,
        **env_args,
    ):
        envs = [
            functools.partial(  # thunk
                self.make_env,
                rank=i,
                env_id=env_id,
                add_timestep=add_timestep,
                seed=seed,
                evaluation=evaluation,
                time_limit=time_limit,
                evaluating=evaluation,
                **env_args,
            )
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
        save_path = Path(
            checkpoint_dir, f"{self.i if self.save_separate else 'checkpoint'}.pt"
        )
        torch.save(dict(step=self.i, **state_dict), save_path)
        print(f"Saved parameters to {save_path}")
        return str(save_path)

    def _restore(self, checkpoint):
        load_path = checkpoint
        state_dict = torch.load(load_path, map_location=self.device)
        self.agent.load_state_dict(state_dict["agent"])
        self.ppo.optimizer.load_state_dict(state_dict["optimizer"])
        self.i = state_dict.get("step", -1) + 1
        # if isinstance(self.envs.venv, VecNormalize):
        #     self.envs.venv.load_state_dict(state_dict["vec_normalize"])
        print(f"Loaded parameters from {load_path}.")

    @abc.abstractmethod
    def get_device(self):
        raise NotImplementedError


class Train(TrainBase):
    def __init__(
        self,
        run_id,
        log_dir: Path,
        save_interval: int,
        num_processes: int,
        num_steps: int,
        save_separate: bool,
        **kwargs,
    ):
        self.save_separate = save_separate
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.run_id = run_id
        self.save_interval = save_interval
        self.log_dir = log_dir
        if log_dir:
            self.writer = SummaryWriter(logdir=str(log_dir))
            self.table = HDF5Store(datapath=str(Path(log_dir, "table")))
        else:
            self.writer = None
            self.table = None
        self.setup(**kwargs, num_processes=num_processes, num_steps=num_steps)
        self.last_save = time.time()  # dummy save

    def run(self):
        for _ in itertools.count():
            for result in self.make_train_iterator():
                if self.writer is not None:
                    self.log_result(result)

                if self.log_dir and self.i % self.save_interval == 0:
                    self._save(str(self.log_dir))
                    self.last_save = time.time()

    def log_result(self, result):
        total_num_steps = (self.i + 1) * self.num_processes * self.num_steps
        for k, v in k_scalar_pairs(**result):
            self.writer.add_scalar(k, v, total_num_steps)

    def get_device(self):
        match = re.search("\d+$", self.run_id)
        if match:
            device_num = int(match.group()) % get_n_gpu()
        else:
            device_num = get_random_gpu()

        return torch.device("cuda", device_num)
