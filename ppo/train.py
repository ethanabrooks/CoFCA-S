from collections import Counter
import functools
import itertools
from pathlib import Path
import re
import sys
import time

import gym
import numpy as np
from tensorboardX import SummaryWriter
import torch
from tqdm import tqdm

from common.atari_wrappers import wrap_deepmind
from common.vec_env.dummy_vec_env import DummyVecEnv
from common.vec_env.subproc_vec_env import SubprocVecEnv
from gridworld_env import SubtasksGridWorld
from ppo.agent import Agent, AgentValues  # noqa
from ppo.storage import RolloutStorage
from ppo.subtasks.wrappers import Wrapper
from ppo.update import PPO
from ppo.utils import get_n_gpu, get_random_gpu
from ppo.wrappers import (
    AddTimestep,
    TransposeImage,
    VecNormalize,
    VecPyTorch,
    VecPyTorchFrameStack,
)

try:
    import dm_control2gym
except ImportError:
    pass


class Train:
    def __init__(
        self,
        num_steps,
        num_processes,
        seed,
        cuda_deterministic,
        cuda,
        max_episode_steps,
        log_dir: Path,
        env_id,
        gamma,
        normalize,
        add_timestep,
        save_interval,
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
        target_success_rate,
        synchronous,
        batch_size,
        run_id,
        save_dir=None,
    ):
        if render_eval and not render:
            eval_interval = 1
        if render:
            ppo_args.update(ppo_epoch=0)
        self.success_reward = success_reward
        save_dir = save_dir or log_dir

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        cuda &= torch.cuda.is_available()
        if cuda and cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        device = 'cpu'
        if cuda:
            device_num = get_random_gpu()
            if run_id:
                match = re.search('\d+$', run_id)
                if match:
                    device_num = int(match.group()) % get_n_gpu()

            device = torch.device('cuda', device_num)
        print('Using device', device)
        self.device = device

        writer = None
        if log_dir:
            writer = SummaryWriter(logdir=str(log_dir))

        torch.set_num_threads(1)

        envs = self.make_vec_envs(
            env_id=env_id,
            seed=seed,
            num_processes=num_processes,
            gamma=(gamma if normalize else None),
            add_timestep=add_timestep,
            render=render,
            synchronous=True if render else synchronous,
            evaluation=False,
        )

        self.agent = self.build_agent(envs=envs, **agent_args)
        rollouts = RolloutStorage(
            num_steps=num_steps,
            num_processes=num_processes,
            obs_space=envs.observation_space,
            action_space=envs.action_space,
            recurrent_hidden_state_size=self.agent.recurrent_hidden_state_size,
        )

        obs = envs.reset()
        rollouts.obs[0].copy_(obs)

        if cuda:
            tick = time.time()
            envs.to(device)
            self.agent.to(device)
            rollouts.to(device)
            print('Values copied to GPU in', time.time() - tick, 'seconds')

        ppo = PPO(agent=self.agent, batch_size=batch_size, **ppo_args)

        counter = Counter()
        start = time.time()
        last_save = start

        if load_path:
            state_dict = torch.load(load_path, map_location=device)
            self.agent.load_state_dict(state_dict['agent'])
            ppo.optimizer.load_state_dict(state_dict['optimizer'])
            start = state_dict.get('step', -1) + 1
            if isinstance(envs.venv, VecNormalize):
                envs.venv.load_state_dict(state_dict["vec_normalize"])
            print(f"Loaded parameters from {load_path}.")

        for j in itertools.count():
            if j % log_interval == 0:
                log_progress = tqdm(total=log_interval, desc="log ")
            if eval_interval and j % eval_interval == 0:
                eval_progress = tqdm(total=eval_interval, desc="eval")
            epoch_counter = self.run_epoch(
                obs=rollouts.obs[0],
                rnn_hxs=rollouts.recurrent_hidden_states[0],
                masks=rollouts.masks[0],
                envs=envs,
                num_steps=num_steps,
                rollouts=rollouts,
                counter=counter,
            )

            with torch.no_grad():
                next_value = self.agent.get_value(
                    rollouts.obs[-1],
                    rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1],
                ).detach()

            rollouts.compute_returns(
                next_value=next_value, use_gae=use_gae, gamma=gamma, tau=tau
            )
            train_results = ppo.update(rollouts)
            rollouts.after_update()

            if save_dir and save_interval and time.time() - last_save >= save_interval:
                last_save = time.time()
                modules = dict(
                    optimizer=ppo.optimizer, agent=self.agent
                )  # type: Dict[str, torch.nn.Module]

                if isinstance(envs.venv, VecNormalize):
                    modules.update(vec_normalize=envs.venv)

                state_dict = {
                    name: module.state_dict() for name, module in modules.items()
                }
                save_path = Path(save_dir, "checkpoint.pt")
                torch.save(dict(step=j, **state_dict), save_path)

                print(f"Saved parameters to {save_path}")

            total_num_steps = (j + 1) * num_processes * num_steps

            mean_success_rate = np.mean(epoch_counter["successes"])
            if target_success_rate and mean_success_rate > target_success_rate:
                print("Finished training with success rate of", mean_success_rate)
                return

            if j % log_interval == 0 and writer is not None:
                end = time.time()
                fps = total_num_steps / (end - start)
                log_values = dict(fps=fps, **epoch_counter, **train_results)
                if writer:
                    for k, v in log_values.items():
                        mean = np.mean(v)
                        if not np.isnan(mean):
                            writer.add_scalar(k, np.mean(v), total_num_steps)

            log_progress.update()

            if eval_interval is not None and j % eval_interval == eval_interval - 1:
                eval_envs = self.make_vec_envs(
                    env_id=env_id,
                    seed=seed + num_processes,
                    num_processes=num_processes,
                    gamma=gamma if normalize else None,
                    add_timestep=add_timestep,
                    evaluation=True,
                    synchronous=True if render_eval else synchronous,
                    render=render_eval,
                )
                eval_envs.to(self.device)

                # vec_norm = get_vec_normalize(eval_envs)
                # if vec_norm is not None:
                #     vec_norm.eval()
                #     vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

                obs = eval_envs.reset()
                eval_recurrent_hidden_states = torch.zeros(
                    num_processes,
                    self.agent.recurrent_hidden_state_size,
                    device=self.device,
                )
                eval_masks = torch.zeros(num_processes, 1, device=self.device)
                eval_counter = Counter()

                eval_values = self.run_epoch(
                    envs=eval_envs,
                    obs=obs,
                    rnn_hxs=eval_recurrent_hidden_states,
                    masks=eval_masks,
                    num_steps=max(num_steps, max_episode_steps)
                    if max_episode_steps
                    else num_steps,
                    rollouts=None,
                    counter=eval_counter,
                )

                eval_envs.close()

                print("Evaluation outcome:")
                if writer is not None:
                    for k, v in eval_values.items():
                        print(f"eval_{k}", np.mean(v))
                        writer.add_scalar(f"eval_{k}", np.mean(v), total_num_steps)

            if eval_interval:
                eval_progress.update()

    def run_epoch(self, obs, rnn_hxs, masks, envs, num_steps, rollouts, counter):
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
            if rollouts is not None:
                rollouts.insert(
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
    def make_env(env_id, seed, rank, add_timestep):
        if env_id.startswith("dm"):
            _, domain, task = env_id.split(".")
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:
            env = gym.make(env_id)

        is_atari = hasattr(gym.envs, "atari") and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv
        )
        if isinstance(env.unwrapped, SubtasksGridWorld):
            env = Wrapper(env)

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

        if (
            envs.observation_space.shape
            and len(envs.observation_space.shape) == 1  # TODO
        ):
            if gamma is None:
                envs = VecNormalize(envs, ret=False)
            else:
                envs = VecNormalize(envs, gamma=gamma)

        envs = VecPyTorch(envs)

        if num_frame_stack is not None:
            envs = VecPyTorchFrameStack(envs, num_frame_stack)
        # elif len(envs.observation_space.shape) == 3:
        #     envs = VecPyTorchFrameStack(envs, 4, self.device)

        return envs
