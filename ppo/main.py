# stdlib
import copy
import glob
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from gym.wrappers import TimeLimit
from tensorboardX import SummaryWriter

from ppo.arg_util import env_wrapper
from ppo.arguments import get_args, get_hsr_args
from ppo.envs import VecPyTorch, make_vec_envs, get_vec_normalize
from ppo.hsr_adaptor import RewardStructure, UnsupervisedDummyVecEnv, UnsupervisedEnv, \
    UnsupervisedSubprocVecEnv, MoveGripperEnv
from ppo.policy import Policy
from ppo.ppo import PPO
from ppo.storage import RolloutStorage


def main(recurrent_policy, num_frames, num_steps, num_processes, seed,
         cuda_deterministic, cuda, log_dir, env_name, gamma, add_timestep,
         save_interval, save_dir, log_interval, eval_interval, use_gae, tau,
         ppo_args, hsr_args, reward_lr):
    algo = 'ppo'

    num_updates = int(num_frames) // num_steps // num_processes

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cuda and torch.cuda.is_available() and cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    writer = SummaryWriter(log_dir=log_dir)
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)

    eval_log_dir = log_dir + "_eval"

    try:
        os.makedirs(eval_log_dir)
    except OSError:
        files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if cuda else "cpu")

    reward_structure = None
    env_args = dict(
        env_name=env_name,
        seed=seed,
        num_processes=num_processes,
        gamma=gamma,
        log_dir=log_dir,
        add_timestep=add_timestep,
        device=device,
        allow_early_resets=False,
        **hsr_args)
    unsupervised = env_name == 'unsupervised'
    if unsupervised:
        sample_env = UnsupervisedEnv(**hsr_args)
        reward_structure = RewardStructure(
            num_processes=num_processes,
            subspace_sizes=sample_env.subspace_sizes,
            reward_function=sample_env.reward_function,
            lr=reward_lr,
        )
        ppo_args.update(reward_structure=reward_structure)
    envs = make_vec_envs(**env_args)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs=dict(recurrent=recurrent_policy))
    actor_critic.to(device)

    agent = PPO(
        actor_critic=actor_critic, unsupervised=unsupervised, **ppo_args)

    rollouts = RolloutStorage(
        num_steps=num_steps,
        num_processes=num_processes,
        obs_shape=envs.observation_space.shape,
        action_space=envs.action_space,
        recurrent_hidden_state_size=actor_critic.recurrent_hidden_state_size,
        reward_structure=reward_structure)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    start = time.time()
    for j in range(num_updates):
        rewards = torch.zeros(num_steps)
        for step in range(num_steps):
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = \
                    actor_critic.act(
                        rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

            # Observe reward and next obs
            obs, reward, done, infos = envs.step(action)
            rewards[step] = torch.mean(reward)

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, use_gae, gamma, tau)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        if unsupervised:
            params = rollouts.reward_params.cpu().detach().numpy()  # type: np.array
            envs.venv.set_reward_params(params)

        rollouts.after_update()

        if j % save_interval == 0 and save_dir != "":
            save_path = os.path.join(save_dir, algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [
                save_model,
                getattr(get_vec_normalize(envs), 'ob_rms', None)
            ]

            torch.save(save_model, os.path.join(save_path, env_name + ".pt"))

        total_num_steps = (j + 1) * num_processes * num_steps

        if j % log_interval == 0:
            end = time.time()
            fps = int(total_num_steps / (end - start))
            writer.add_scalar('rewards', float(rewards.mean()), j)
            writer.add_scalar('fps', fps, j)
            writer.add_scalar('value loss', value_loss, j)
            writer.add_scalar('action loss', action_loss, j)
            writer.add_scalar('entropy', dist_entropy, j)
            print(
                f"Updates {j}, num timesteps {total_num_steps}, FPS {fps}, reward "
                f"{rewards.mean()}")
            if unsupervised:
                with Path(log_dir, 'params.npy').open('a') as f:
                    np.savetxt(f, params[0].reshape(1, -1))

        if eval_interval is not None and j % eval_interval == 0:
            env_args.update(seed=seed + num_processes + j,
                            record_path=Path(log_dir, 'eval.mp4'))
            eval_envs = make_vec_envs(**env_args)

            # TODO: should this be here?
            vec_norm = get_vec_normalize(eval_envs)
            if vec_norm is not None:
                vec_norm.eval()
                vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

            obs = eval_envs.reset()
            eval_recurrent_hidden_states = torch.zeros(
                num_processes,
                actor_critic.recurrent_hidden_state_size,
                device=device)
            eval_masks = torch.zeros(num_processes, 1, device=device)

            eval_episode_rewards = torch.zeros(num_steps)
            for step in range(num_steps):
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs,
                        eval_recurrent_hidden_states,
                        eval_masks,
                        deterministic=True)

                # Obser reward and next obs
                obs, reward, done, infos = eval_envs.step(action)
                eval_episode_rewards[step] = torch.mean(reward)

                eval_masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])

            eval_envs.close()

            mean_eval_reward = float(torch.mean(eval_episode_rewards))
            print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
                len(eval_episode_rewards), mean_eval_reward))
            writer.add_scalar('eval reward', mean_eval_reward, j)


def cli():
    main(**get_args())


def hsr_cli():
    env_wrapper(main)(**get_hsr_args())


if __name__ == "__main__":
    hsr_cli()
