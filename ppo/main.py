# stdlib
import copy
import glob
import os
import time
from collections import deque
import sys

import numpy as np
import torch
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from environments.hsr import MoveGripperEnv
from gym.wrappers import TimeLimit
from scripts.hsr import env_wrapper

from ppo.arguments import get_args, get_hsr_args
from ppo.envs import make_vec_envs, VecPyTorch
from ppo.hsr_wrapper import UnsupervisedEnv, UnsupervisedDummyVecEnv, \
    RewardStructure, UnsupervisedSubprocVecEnv
from ppo.model import Policy
from ppo.ppo import PPO
from ppo.storage import RolloutStorage
from ppo.utils import get_vec_normalize
from ppo.visualize import visdom_plot


def main(recurrent_policy, num_frames, num_steps, num_processes, seed,
         cuda_deterministic, cuda, log_dir, env_name, gamma, add_timestep,
         save_interval, save_dir, log_interval, eval_interval, use_gae, tau,
         vis_interval, visdom_args, ppo_args, env_args):
    algo = 'ppo'

    num_updates = int(num_frames) // num_steps // num_processes

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cuda and torch.cuda.is_available() and cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

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

    vis = all(visdom_args.values())
    if vis:
        from visdom import Visdom
        viz = Visdom(**visdom_args)
        win = None

    reward_structure = None
    unsupervised = env_name == 'unsupervised'
    if unsupervised:

        def make_env(_seed):
            env = TimeLimit(
                UnsupervisedEnv(**env_args), max_episode_steps=num_steps)
            env.seed(_seed)
            return env

        sample_env = make_env(0).env
        reward_structure = RewardStructure(
            num_processes=num_processes,
            subspace_sizes=sample_env.subspace_sizes,
            reward_function=sample_env.reward_function)
        ppo_args.update(reward_params=reward_structure.reward_params)

        env_fns = [lambda: make_env(s + seed) for s in range(num_processes)]

        if sys.platform == 'darwin' or num_processes == 1:
            envs = UnsupervisedSubprocVecEnv(env_fns)
        else:
            envs = UnsupervisedDummyVecEnv(env_fns)
        envs = VecPyTorch(envs, device=device)

    elif env_name == 'move_gripper':

        def make_env():
            MoveGripperEnv(**env_args)

        envs = VecPyTorch(
            DummyVecEnv([make_env()] * num_processes), device=device)
    else:
        envs = make_vec_envs(env_name, seed, num_processes, gamma, log_dir,
                             add_timestep, device, False)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': recurrent_policy})
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

    episode_rewards = deque(maxlen=10)

    start = time.time()
    for j in range(num_updates):
        for step in range(num_steps):
            # Sample actions.add_argument_group('env_args')
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = \
                    actor_critic.act(
                        rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

            # Observe reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

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
            params = rollouts.reward_params.detach().numpy()
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

        if j % log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: "
                "mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

        if (eval_interval is not None and len(episode_rewards) > 1
                and j % eval_interval == 0):
            eval_envs = make_vec_envs(env_name, seed + num_processes,
                                      num_processes, gamma, eval_log_dir,
                                      add_timestep, device, True)

            vec_norm = get_vec_normalize(eval_envs)
            if vec_norm is not None:
                vec_norm.eval()
                vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

            eval_episode_rewards = []

            obs = eval_envs.reset()
            eval_recurrent_hidden_states = torch.zeros(
                num_processes,
                actor_critic.recurrent_hidden_state_size,
                device=device)
            eval_masks = torch.zeros(num_processes, 1, device=device)

            while len(eval_episode_rewards) < 10:
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs,
                        eval_recurrent_hidden_states,
                        eval_masks,
                        deterministic=True)

                # Obser reward and next obs
                obs, reward, done, infos = eval_envs.step(action)

                eval_masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                for info in infos:
                    if 'episode' in info.keys():
                        eval_episode_rewards.append(info['episode']['r'])

            eval_envs.close()

            print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
                len(eval_episode_rewards), np.mean(eval_episode_rewards)))

        if vis and j % vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, log_dir, env_name, algo,
                                  num_frames)
            except IOError:
                pass


def cli():
    main(**get_args())


def hsr_cli():
    env_wrapper(main)(**get_hsr_args())


if __name__ == "__main__":
    hsr_cli()
