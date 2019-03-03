import itertools
from pathlib import Path
import time

from gym.spaces import Discrete
import numpy as np
from tensorboardX import SummaryWriter
import torch

from ppo.envs import VecNormalize, make_vec_envs
from ppo.policy import Policy
from ppo.storage import RolloutStorage, TasksRolloutStorage
from ppo.task_generator import TaskGenerator
from ppo.update import PPO
from utils import onehot, space_to_size


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
          num_processes,
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
    sample_env = make_env(seed=seed, rank=0, evaluation=False).unwrapped
    num_tasks = sample_env.task_space.n
    if train_tasks:
        num_processes = 1
    else:
        num_processes = num_tasks

    if log_dir:
        plt.switch_backend('agg')
        xlim, ylim = sample_env.desc.shape

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
    if eval_interval:
        eval_envs = make_vec_envs(
            seed=seed + num_processes,
            make_env=make_env,
            num_processes=num_tasks,
            gamma=_gamma,
            device=device,
            train_tasks=train_tasks,
            normalize=normalize,
            eval=True,
        )

    actor_critic = Policy(
        envs.observation_space, envs.action_space, network_args=network_args)

    gan = None
    tasks_data = []
    last_index = 0
    if train_tasks:
        gan = TaskGenerator(
            task_size=sample_env.task_space.n,
            **{k.replace('gan_', ''): v
               for k, v in tasks_args.items()})

        tasks_to_train = torch.tensor(gan.sample(1), dtype=torch.float)
        envs.unwrapped.set_task_dist(0, onehot(int(tasks_to_train), num_tasks))
        for i in range(1, num_processes):
            envs.unwrapped.set_task_dist(
                i, onehot(int(tasks_to_train) + 1 % num_tasks, num_tasks))

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
    time_step_counter = np.zeros(num_processes)
    episode_rewards = []
    time_steps = []
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

    if train_tasks:
        tasks = gan.sample(num_processes)
        # tasks = np.arange(num_processes)
        for i, task in enumerate(tasks):
            envs.unwrapped.set_task_dist(i, onehot(task, num_tasks))

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    if train_tasks:
        tasks = torch.tensor(envs.unwrapped.get_tasks())
        importance_weights = gan.importance_weight(tasks)
        rollouts.tasks[0].copy_(tasks.view(rollouts.tasks[0].size()))
        rollouts.importance_weighting[0].copy_(
            importance_weights.view(rollouts.importance_weighting[0].size()))

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
            obs, rewards, dones, infos = envs.step(actions)

            # track rewards
            rewards_counter += rewards.numpy()
            time_step_counter += 1
            episode_rewards.append(rewards_counter[dones])
            time_steps.append(time_step_counter[dones])
            rewards_counter[dones] = 0
            time_step_counter[dones] = 0

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in dones])
            if train_tasks:
                tasks = torch.tensor(envs.unwrapped.get_tasks())
                importance_weights = gan.importance_weight(tasks)
                rollouts.insert(
                    obs=obs,
                    recurrent_hidden_states=recurrent_hidden_states,
                    actions=actions,
                    action_log_probs=action_log_probs,
                    values=values,
                    rewards=rewards,
                    masks=masks,
                    task=tasks,
                    importance_weighting=importance_weights,
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

        train_results, task_stuff = agent.update(
            rollouts, tasks_to_train=tasks_to_train)
        if train_tasks:
            tasks = gan.sample(num_processes)
            # tasks = np.arange(num_processes)
            for i, task in enumerate(tasks):
                envs.unwrapped.set_task_dist(i, onehot(task, num_tasks))

            tasks_trained, task_returns, gradient_sums = task_stuff
            tasks_trained = sample_env.task_states[tasks_trained.int().numpy()]
            tasks_data.extend(
                [(x, y, r, g)
                 for x, y, r, g in zip(*sample_env.decode(tasks_trained),
                                       task_returns, gradient_sums)])

        if train_tasks:
            tasks_to_train = torch.tensor(gan.sample(1), dtype=torch.float)
            envs.unwrapped.set_task_dist(
                0, onehot(int(tasks_to_train), num_tasks))
            for i in range(1, num_processes):
                envs.unwrapped.set_task_dist(
                    i, onehot(int(tasks_to_train) + 1 % num_tasks, num_tasks))

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
            time_steps = np.concatenate(time_steps)
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
                writer.add_scalar('time steps', np.mean(time_steps),
                                  total_num_steps)
                writer.add_scalar('num tasks', len(tasks_data),
                                  total_num_steps)
                for k, v in train_results.items():
                    if v.dim() == 0:
                        writer.add_scalar(k, v, total_num_steps)

                if train_tasks:
                    x, y, rewards, gradient = zip(*tasks_data)

                    def plot(c, text, x=x, y=y):
                        fig = plt.figure()
                        x_noise = (np.random.rand(len(x)) - .5) * .9
                        y_noise = (np.random.rand(len(y)) - .5) * .9
                        sc = plt.scatter(
                            x + x_noise,
                            y + y_noise,
                            c=c,
                            cmap=cm.hot,
                            alpha=.1)
                        plt.colorbar(sc)
                        axes = plt.axes()
                        axes.set_xlim(-.5, xlim - .5)
                        axes.set_ylim(-.5, ylim - .5)
                        plt.subplots_adjust(.15, .15, .95, .95)
                        writer.add_figure(text, fig, total_num_steps)
                        plt.close(fig)

                    fig = plt.figure()
                    probs = np.zeros(sample_env.desc.shape)
                    probs[sample_env.decode(
                        sample_env.task_states)] = gan.probs().detach()
                    im = plt.imshow(probs, origin='lower')
                    plt.colorbar(im)
                    writer.add_figure('probs', fig, total_num_steps)
                    plt.close()

                    plot(rewards, 'rewards')
                    plot(gradient, 'gradients')

                    x, y, rewards, gradient = zip(*tasks_data[last_index:])
                    last_index = len(tasks_data)
            episode_rewards = []
            time_steps = []

        if eval_interval is not None and j % eval_interval == 0:
            eval_rewards = np.zeros(num_tasks)
            eval_time_steps = np.zeros(num_tasks)
            eval_done = np.zeros(num_tasks)

            obs = eval_envs.reset()
            eval_recurrent_hidden_states = torch.zeros(
                num_tasks,
                actor_critic.recurrent_hidden_state_size,
                device=device)
            eval_masks = torch.zeros(num_tasks, 1, device=device)

            while not np.all(eval_done):
                with torch.no_grad():
                    _, actions, _, eval_recurrent_hidden_states = actor_critic.act(
                        inputs=obs,
                        rnn_hxs=eval_recurrent_hidden_states,
                        masks=eval_masks,
                        deterministic=True)

                # Observe reward and next obs
                obs, rewards, dones, infos = eval_envs.step(actions)
                not_done = eval_done == 0
                eval_rewards[not_done] += rewards.numpy()[not_done]
                eval_time_steps[not_done] += 1
                eval_done[dones] = 1

                eval_masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in dones])

            if log_dir:
                writer.add_scalar('eval return', np.mean(eval_rewards),
                                  total_num_steps)
                writer.add_scalar('eval time steps', np.mean(eval_time_steps),
                                  total_num_steps)

            print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
                num_tasks, np.mean(eval_rewards)))
