# stdlib
import copy
import itertools
import os
import time
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter

# first party
from environments.hsr import Observation
from ppo.arguments import build_parser, get_hsr_parser, get_unsupervised_parser
from ppo.envs import make_vec_envs
from ppo.gan import GAN
from ppo.hsr_adapter import UnsupervisedEnv
from ppo.policy import Policy
from ppo.ppo import PPO
from ppo.storage import RolloutStorage
from ppo.utils import get_vec_normalize
from scripts.hsr import env_wrapper, parse_groups

# third party


def main(recurrent_policy,
         num_frames,
         num_steps,
         num_processes,
         seed,
         cuda_deterministic,
         cuda,
         log_dir: Path,
         env_name,
         gamma,
         normalize,
         add_timestep,
         save_interval,
         load_path,
         log_interval,
         eval_interval,
         use_gae,
         tau,
         ppo_args,
         network_args,
         max_steps=None,
         env_args=None,
         unsupervised_args=None):
    algo = 'ppo'

    if num_frames:
        updates = range(int(num_frames) // num_steps // num_processes)
    else:
        updates = itertools.count()

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cuda and torch.cuda.is_available() and cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    eval_log_dir = None
    if log_dir:
        writer = SummaryWriter(log_dir=str(log_dir))
        eval_log_dir = log_dir.joinpath("eval")

        for _dir in [log_dir, eval_log_dir]:
            try:
                _dir.mkdir()
            except OSError:
                for f in _dir.glob('*.monitor.csv'):
                    f.unlink()

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if cuda else "cpu")

    unsupervised = unsupervised_args is not None

    _gamma = gamma if normalize else None
    envs = make_vec_envs(
        env_name=env_name,
        seed=seed,
        num_processes=num_processes,
        gamma=_gamma,
        log_dir=log_dir,
        add_timestep=add_timestep,
        device=device,
        max_steps=max_steps,
        allow_early_resets=False,
        env_args=env_args)

    obs = envs.reset()

    gan = None
    if unsupervised:
        sample_env = UnsupervisedEnv(**env_args)
        gan = GAN(
            goal_size=3,
            **{k.replace('gan_', ''): v
               for k, v in unsupervised_args.items()})

        def substitute_goal(_obs, _goals):
            split = torch.split(_obs, sample_env.subspace_sizes, dim=1)
            replace = Observation(*split)._replace(goal=_goals)
            return torch.cat(replace, dim=1)

        goals = gan.sample(num_processes)
        for i, goal in enumerate(goals):
            envs.unwrapped.set_goal(goal.detach().numpy(), i)
        obs = substitute_goal(obs, goals)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        network_args=network_args)

    if load_path:
        state_dict = torch.load(load_path)
        if unsupervised:
            gan.load_state_dict(state_dict['gan'])
        actor_critic.load_state_dict(state_dict['actor_critic'])

    actor_critic.to(device)
    if unsupervised:
        gan.to(device)

    agent = PPO(actor_critic=actor_critic, gan=gan, **ppo_args)

    rollouts = RolloutStorage(
        num_steps=num_steps,
        num_processes=num_processes,
        obs_shape=envs.observation_space.shape,
        action_space=envs.action_space,
        recurrent_hidden_state_size=actor_critic.recurrent_hidden_state_size,
    )

    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    rewards_counter = np.zeros(num_processes)
    episode_rewards = []

    start = time.time()
    for j in updates:
        for step in range(num_steps):
            # Sample actions.add_argument_group('env_args')
            with torch.no_grad():
                values, actions, action_log_probs, recurrent_hidden_states = \
                    actor_critic.act(
                        inputs=rollouts.obs[step],
                        rnn_hxs=rollouts.recurrent_hidden_states[step],
                        masks=rollouts.masks[step])

            # Observe reward and next obs
            obs, rewards, done, infos = envs.step(actions)

            if unsupervised:
                for i, _done in enumerate(done):
                    if _done:
                        goal = gan.sample(1)
                        envs.unwrapped.set_goal(goal.detach().numpy(), i)
                        goals[i] = goal
                obs = substitute_goal(obs, goals)

            # track rewards
            rewards_counter += rewards
            episode_rewards.append(rewards_counter[done])
            rewards_counter[done] = 0

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            rollouts.insert(
                obs=obs,
                recurrent_hidden_states=recurrent_hidden_states,
                actions=actions,
                action_log_probs=action_log_probs,
                values=values,
                rewards=rewards,
                masks=masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                inputs=rollouts.obs[-1],
                rnn_hxs=rollouts.recurrent_hidden_states[-1],
                masks=rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value=next_value,
                                 use_gae=use_gae,
                                 gamma=gamma,
                                 tau=tau)
        train_results = agent.update(rollouts)
        rollouts.after_update()

        if j % save_interval == 0 and log_dir is not None:
            models = dict(actor_critic=actor_critic)  # type: Dict[str, nn.Module]
            if unsupervised:
                models.update(gan=gan)
            state_dict = {name: model.state_dict() for name, model in models.items()}
            save_path = Path(log_dir, 'checkpoint.pt')
            torch.save(state_dict, save_path)


        total_num_steps = (j + 1) * num_processes * num_steps

        if j % log_interval == 0:
            end = time.time()
            fps = int(total_num_steps / (end - start))
            episode_rewards = np.concatenate(episode_rewards)
            if episode_rewards.size > 0:
                print(
                    f"Updates {j}, num timesteps {total_num_steps}, FPS {fps} \n "
                    f"Last {len(episode_rewards)} training episodes: " +
                    "mean/median reward {:.2f}/{:.2f}, min/max reward {:.2f}/{"
                    ":.2f}\n".format(
                        np.mean(episode_rewards), np.median(episode_rewards),
                        np.min(episode_rewards), np.max(episode_rewards)))
            if log_dir:
                writer.add_scalar('return', np.mean(episode_rewards), j)
                for k, v in train_results.items():
                    if np.isscalar(v):
                        writer.add_scalar(k.replace('_', ' '), v, j)
            episode_rewards = []

        if eval_interval is not None and j % eval_interval == eval_interval - 1:
            eval_envs = make_vec_envs(
                env_name=env_name,
                seed=seed + num_processes,
                num_processes=num_processes,
                gamma=_gamma,
                log_dir=eval_log_dir,
                add_timestep=add_timestep,
                device=device,
                max_steps=max_steps,
                env_args=env_args,
                allow_early_resets=True)

            # vec_norm = get_vec_normalize(eval_envs)
            # if vec_norm is not None:
            #     vec_norm.eval()
            #     vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

            eval_episode_rewards = []

            obs = eval_envs.reset()
            eval_recurrent_hidden_states = torch.zeros(
                num_processes,
                actor_critic.recurrent_hidden_state_size,
                device=device)
            eval_masks = torch.zeros(num_processes, 1, device=device)

            while len(eval_episode_rewards) < 10:
                print('.', end='')
                with torch.no_grad():
                    _, actions, _, eval_recurrent_hidden_states = actor_critic.act(
                        inputs=obs,
                        rnn_hxs=eval_recurrent_hidden_states,
                        masks=eval_masks,
                        deterministic=True)

                # Observe reward and next obs
                obs, rewards, done, infos = eval_envs.step(actions)

                eval_masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                for info in infos:
                    if 'episode' in info.keys():
                        eval_episode_rewards.append(info['episode']['r'])

            eval_envs.close()

            print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
                len(eval_episode_rewards), np.mean(eval_episode_rewards)))


def cli():
    main(**parse_groups(build_parser()))


def hsr_cli():
    parser = get_hsr_parser()
    env_wrapper(main)(**parse_groups(parser))


def unsupervised_cli():
    parser = get_unsupervised_parser()
    args_dict = parse_groups(parser)
    args_dict.update(env_name='unsupervised')
    env_wrapper(main)(**args_dict)


if __name__ == "__main__":
    hsr_cli()
