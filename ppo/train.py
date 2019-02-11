import itertools
from pathlib import Path
import time

from gym.spaces import Discrete
import numpy as np
from tensorboardX import SummaryWriter
import torch

from ppo.envs import VecNormalize, make_vec_envs
from ppo.gan import GAN
from ppo.policy import Policy
from ppo.storage import RolloutStorage, UnsupervisedRolloutStorage
from ppo.update import PPO
from utils import space_to_size


def train(num_frames,
          num_steps,
          num_processes,
          seed,
          cuda_deterministic,
          cuda,
          log_dir: Path,
          make_env,
          gamma,
          normalize,
          save_interval,
          load_path,
          log_interval,
          eval_interval,
          use_gae,
          tau,
          ppo_args,
          network_args,
          render,
          unsupervised_args=None):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cuda and torch.cuda.is_available() and cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if log_dir:
        writer = SummaryWriter(log_dir=str(log_dir))
        print(f'Logging to {log_dir}')
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

    if render:
        num_processes = 1
    envs = make_vec_envs(
        make_env=make_env,
        seed=seed,
        num_processes=num_processes,
        gamma=gamma,
        device=device,
        unsupervised=unsupervised,
        normalize=normalize)

    actor_critic = Policy(
        envs.observation_space, envs.action_space, network_args=network_args)

    gan = None
    if unsupervised:
        sample_env = make_env(seed=seed, rank=0).unwrapped
        gan = GAN(
            goal_space=sample_env.goal_space,
            **{k.replace('gan_', ''): v
               for k, v in unsupervised_args.items()})

        samples, goals, importance_weightings = gan.sample(num_processes)
        for i, goal in enumerate(goals):
            goal = goal.detach().numpy()
            envs.unwrapped.set_goal(goal, i)

        if isinstance(sample_env.goal_space, Discrete):
            goal_size = 1
        else:
            goal_size = space_to_size(sample_env.goal_space)
        rollouts = UnsupervisedRolloutStorage(
            num_steps=num_steps,
            num_processes=num_processes,
            obs_shape=envs.observation_space.shape,
            action_space=envs.action_space,
            recurrent_hidden_state_size=actor_critic.
            recurrent_hidden_state_size,
            goal_size=goal_size)

    else:
        rollouts = RolloutStorage(
            num_steps=num_steps,
            num_processes=num_processes,
            obs_shape=envs.observation_space.shape,
            action_space=envs.action_space,
            recurrent_hidden_state_size=actor_critic.
            recurrent_hidden_state_size,
        )

    agent = PPO(actor_critic=actor_critic, gan=gan, **ppo_args)

    rewards_counter = np.zeros(num_processes)
    episode_rewards = []
    last_save = time.time()

    start = 0
    if load_path:
        state_dict = torch.load(load_path)
        if unsupervised:
            gan.load_state_dict(state_dict['gan'])
        actor_critic.load_state_dict(state_dict['actor_critic'])
        agent.optimizer.load_state_dict(state_dict['optimizer'])
        start = state_dict.get('step', -1) + 1
        if isinstance(envs.venv, VecNormalize):
            envs.venv.load_state_dict(state_dict['vec_normalize'])
        print(f'Loaded parameters from {load_path}.')

    if num_frames:
        updates = range(start, int(num_frames) // num_steps // num_processes)
    else:
        updates = itertools.count(start)

    actor_critic.to(device)
    if unsupervised:
        gan.to(device)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    if unsupervised:
        rollouts.goals[0].copy_(samples.view(-1, 1))
        rollouts.importance_weighting[0].copy_(importance_weightings)
    rollouts.to(device)

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

            if render:
                envs.render()
                time.sleep(.5)

            # Observe reward and next obs
            obs, rewards, done, infos = envs.step(actions)

            if unsupervised:
                for i, _done in enumerate(done):
                    if _done:
                        sample, goal, importance_weighting = gan.sample(1)
                        goal = goal.detach().numpy()
                        envs.unwrapped.set_goal(goal, i)
                        samples[i] = sample
                        importance_weightings[i] = importance_weighting

            # track rewards
            rewards_counter += rewards.numpy()
            episode_rewards.append(rewards_counter[done])
            rewards_counter[done] = 0

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            if unsupervised:
                rollouts.insert(
                    obs=obs,
                    recurrent_hidden_states=recurrent_hidden_states,
                    actions=actions,
                    action_log_probs=action_log_probs,
                    values=values,
                    rewards=rewards,
                    masks=masks,
                    goal=samples,
                    importance_weighting=importance_weightings,
                )
            else:
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

        rollouts.compute_returns(
            next_value=next_value, use_gae=use_gae, gamma=gamma, tau=tau)

        train_results = agent.update(rollouts)
        rollouts.after_update()
        total_num_steps = (j + 1) * num_processes * num_steps

        if all(
            [log_dir, save_interval,
             time.time() - last_save >= save_interval]):
            last_save = time.time()
            modules = dict(
                optimizer=agent.optimizer,
                actor_critic=actor_critic)  # type: Dict[str, torch.nn.Module]

            if isinstance(envs.venv, VecNormalize):
                modules.update(vec_normalize=envs.venv)

            if unsupervised:
                modules.update(gan=gan)
            state_dict = {
                name: module.state_dict()
                for name, module in modules.items()
            }
            save_path = Path(log_dir, 'checkpoint.pt')
            torch.save(dict(step=j, **state_dict), save_path)

            print(f'Saved parameters to {save_path}')

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
                print(f'Writing log data to {log_dir}.')
                writer.add_scalar('fps', fps, total_num_steps)
                writer.add_scalar('return', np.mean(episode_rewards),
                                  total_num_steps)
                for k, v in train_results.items():
                    if v.dim() == 0:
                        writer.add_scalar(k, v, total_num_steps)
            episode_rewards = []

        if eval_interval is not None and j % eval_interval == eval_interval - 1:
            eval_envs = make_vec_envs(
                seed=seed + num_processes,
                make_env=make_env,
                num_processes=num_processes,
                gamma=gamma,
                device=device,
                unsupervised=unsupervised)

            eval_episode_rewards = []

            obs = eval_envs.reset()
            eval_recurrent_hidden_states = torch.zeros(
                num_processes,
                actor_critic.recurrent_hidden_state_size,
                device=device)
            eval_masks = torch.zeros(num_processes, 1, device=device)

            while len(eval_episode_rewards) < 10:
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
