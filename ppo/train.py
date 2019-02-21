import itertools
from pathlib import Path
import time

from gym.spaces import Discrete
import numpy as np
from tensorboardX import SummaryWriter
import torch

from ppo.env_adapter import TasksHSREnv
from ppo.envs import VecNormalize, make_vec_envs
from ppo.policy import Policy
from ppo.storage import RolloutStorage, TasksRolloutStorage
from ppo.task_generator import TaskGenerator
from ppo.update import PPO
from utils import space_to_size


def train(num_frames,
          num_steps,
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
          synchronous,
          tasks_args=None):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cuda and torch.cuda.is_available() and cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if log_dir:
        import matplotlib.pyplot as plt

        import matplotlib.cm as cm
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

    train_tasks = tasks_args is not None
    sample_env = make_env(seed=seed, rank=0, eval=False).unwrapped

    num_processes = sample_env.task_space.n

    if log_dir:
        plt.switch_backend('agg')
        axes = plt.axes()
        xlim, ylim = sample_env.desc.shape
        axes.set_xlim(-.5, xlim + .5)
        axes.set_ylim(-.5, ylim + .5)

    _gamma = gamma if normalize else None
    envs = make_vec_envs(
        make_env=make_env,
        seed=seed,
        num_processes=num_processes,
        gamma=_gamma,
        device=device,
        train_tasks=train_tasks,
        normalize=normalize,
        synchronous=synchronous,
        eval=False)

    actor_critic = Policy(
        envs.observation_space, envs.action_space, network_args=network_args)

    gan = None
    tasks_data = []
    last_index = 0
    if train_tasks:
        assert sample_env.task_space.n == num_processes
        gan = TaskGenerator(
            task_space=sample_env.task_space,
            **{k.replace('gan_', ''): v
               for k, v in tasks_args.items()})

        # samples, tasks, importance_weightings = gan.sample(num_processes)
        samples = tasks = torch.arange(sample_env.task_space.n)
        importance_weightings = torch.zeros_like(tasks)
        for i, task in enumerate(tasks):
            task = task.detach().numpy()
            # envs.unwrapped.set_task(task, i)

        if isinstance(sample_env.task_space, Discrete):
            task_size = 1
        else:
            task_size = space_to_size(sample_env.task_space)
        rollouts = TasksRolloutStorage(
            num_steps=num_steps,
            num_processes=num_processes,
            obs_shape=envs.observation_space.shape,
            action_space=envs.action_space,
            recurrent_hidden_state_size=actor_critic.
            recurrent_hidden_state_size,
            task_size=task_size)

    else:
        rollouts = RolloutStorage(
            num_steps=num_steps,
            num_processes=num_processes,
            obs_shape=envs.observation_space.shape,
            action_space=envs.action_space,
            recurrent_hidden_state_size=actor_critic.
            recurrent_hidden_state_size,
        )

    agent = PPO(actor_critic=actor_critic, task_generator=gan, **ppo_args)

    rewards_counter = np.zeros(num_processes)
    episode_rewards = []
    last_save = time.time()

    start = 0
    if load_path:
        state_dict = torch.load(load_path)
        if train_tasks:
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
    if train_tasks:
        gan.to(device)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    if train_tasks:
        rollouts.tasks[0].copy_(samples.view(-1, 1))
        # rollouts.importance_weighting[0].copy_(importance_weightings)
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

            # track rewards
            rewards_counter += rewards.numpy()
            episode_rewards.append(rewards_counter[done])
            rewards_counter[done] = 0

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            if train_tasks:
                rollouts.insert(
                    obs=obs,
                    recurrent_hidden_states=recurrent_hidden_states,
                    actions=actions,
                    action_log_probs=action_log_probs,
                    values=values,
                    rewards=rewards,
                    masks=masks,
                    task=samples,
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

        train_results, tasks_trained, returns, gradient_sums = agent.update(
            rollouts)
        tasks_trained = sample_env.task_states[torch.cat(
            tasks_trained).int().numpy()]
        l = [(x, y, r, g) for x, y, r, g in zip(
            *sample_env.decode(tasks_trained), returns, gradient_sums)]
        tasks_data.extend(l)

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

            if train_tasks:
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
                writer.add_scalar('num tasks', len(tasks_data),
                                  total_num_steps)
                for k, v in train_results.items():
                    if v.dim() == 0:
                        writer.add_scalar(k, v, total_num_steps)
                # if train_tasks:
                #     writer.add_histogram('gan probs', np.array(tasks_trained),
                #                          total_num_steps)

                x, y, rewards, gradient = zip(*tasks_data)

                def plot(c, text):
                    fig = plt.figure()
                    x_noise = (np.random.rand(len(x)) - .5) * .9
                    y_noise = (np.random.rand(len(y)) - .5) * .9
                    sc = plt.scatter(
                        x + x_noise, y + y_noise, c=c, cmap=cm.hot, alpha=.1)
                    plt.colorbar(sc)
                    plt.subplots_adjust(.15, .15, .95, .95)
                    writer.add_figure(text, fig, total_num_steps)
                    plt.close(fig)

                plot(rewards, 'rewards')
                plot(gradient, 'gradients')

                x, y, rewards, gradient = zip(*tasks_data[last_index:])
                last_index = len(tasks_data)
                plot(rewards, 'new rewards')
                plot(gradient, 'new gradients')
            episode_rewards = []

        if eval_interval is not None and j % eval_interval == 0:
            eval_envs = make_vec_envs(
                seed=seed + num_processes,
                make_env=make_env,
                num_processes=num_processes,
                gamma=_gamma,
                device=device,
                train_tasks=train_tasks,
                normalize=normalize,
                eval=True,
            )

            eval_episode_rewards = []
            eval_rewards_counter = np.zeros(num_processes)

            obs = eval_envs.reset()
            eval_recurrent_hidden_states = torch.zeros(
                num_processes,
                actor_critic.recurrent_hidden_state_size,
                device=device)
            eval_masks = torch.zeros(num_processes, 1, device=device)

            while len(eval_episode_rewards) < num_processes:
                with torch.no_grad():
                    _, actions, _, eval_recurrent_hidden_states = actor_critic.act(
                        inputs=obs,
                        rnn_hxs=eval_recurrent_hidden_states,
                        masks=eval_masks,
                        deterministic=True)

                # Observe reward and next obs
                obs, rewards, done, infos = eval_envs.step(actions)
                eval_rewards_counter += rewards.numpy()
                if done.any():
                    eval_episode_rewards.append(eval_rewards_counter[done])
                    eval_rewards_counter[done] = 0

                eval_masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])

            eval_episode_rewards = np.concatenate(eval_episode_rewards)
            if log_dir:
                writer.add_scalar('eval return', np.mean(eval_episode_rewards),
                                  total_num_steps)

            eval_envs.close()

            print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
                len(eval_episode_rewards), np.mean(eval_episode_rewards)))
