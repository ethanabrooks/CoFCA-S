import csv
from io import StringIO
import itertools
from pathlib import Path
import subprocess
import time

from gym.spaces import Discrete
import numpy as np
from tensorboardX import SummaryWriter
import torch

from ppo.env_adapter import AutoCurriculumHSREnv, GridWorld
from ppo.envs import VecNormalize, make_vec_envs
from ppo.policy import Policy
from ppo.storage import RolloutStorage, TasksRolloutStorage
from ppo.task_generator import GoalGAN, RewardBasedTaskGenerator, TaskGenerator
from ppo.update import PPO
from utils import ReplayBuffer, space_to_size


def get_freer_gpu():
    nvidia_smi = subprocess.check_output(
        'nvidia-smi --format=csv --query-gpu=memory.free'.split(),
        universal_newlines=True)
    free_memory = [
        float(x[0].split()[0])
        for i, x in enumerate(csv.reader(StringIO(nvidia_smi))) if i > 0
    ]
    return np.argmax(free_memory)


def train(
        num_frames,
        deterministic_eval,
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
        solved,
        num_solved,
        task_history,
        tasks_args=None,
        reward_based_task_args=None,
        gan_args=None,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cuda &= torch.cuda.is_available()
    if cuda and cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    sample_env = make_env(seed=seed, rank=0, evaluation=False)

    if log_dir:
        import matplotlib.pyplot as plt

        writer = SummaryWriter(log_dir=str(log_dir))
        print(f'Logging to {log_dir}')
        eval_log_dir = log_dir.joinpath("eval")

        for _dir in [log_dir, eval_log_dir]:
            try:
                _dir.mkdir()
            except OSError:
                for f in _dir.glob('*.monitor.csv'):
                    f.unlink()

        if isinstance(sample_env.unwrapped, AutoCurriculumHSREnv):
            for i, image in enumerate(sample_env.unwrapped.start_images):
                fig = plt.figure()
                plt.imshow(image)
                name = f'start state {i}'
                writer.add_figure(name, fig, 0)
                plt.close()

    torch.set_num_threads(1)
    device = torch.device(f"cuda:{get_freer_gpu()}" if cuda else "cpu")
    print('Using device:', device)

    train_tasks = tasks_args is not None

    num_tasks = None
    if train_tasks:
        task_space = sample_env.unwrapped.task_space
        num_tasks = task_space.n

    if log_dir:
        plt.switch_backend('agg')

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
    actor_critic = Policy(envs.observation_space, envs.action_space,
                          **network_args)

    task_generator = None
    if train_tasks:
        last_n_tasks = ReplayBuffer(maxlen=task_history)
        task_counts = np.zeros(num_tasks)
        last_gradient = torch.zeros(num_tasks).to(device)

        sampling_strategy = tasks_args['sampling_strategy']
        if sampling_strategy in ['reward-variance', 'reward-range']:
            task_generator = RewardBasedTaskGenerator(
                task_size=num_tasks, **reward_based_task_args, **tasks_args)
        elif sampling_strategy == 'goal-gan':
            task_generator = GoalGAN(**tasks_args, **gan_args)
        else:
            task_generator = TaskGenerator(task_size=num_tasks, **tasks_args)
        task_generator = task_generator.to(device)

        if isinstance(task_space, Discrete):
            task_size = 1
        else:
            task_size = space_to_size(task_space)
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

    if eval_interval:
        num_eval = sample_env.unwrapped.num_eval if train_tasks else num_processes
        eval_envs = make_vec_envs(
            seed=seed + num_processes,
            make_env=make_env,
            num_processes=num_eval,
            gamma=_gamma,
            device=device,
            train_tasks=train_tasks,
            normalize=normalize,
            synchronous=synchronous,
            eval=True,
        )

    agent = PPO(
        actor_critic=actor_critic, task_generator=task_generator, **ppo_args)

    rewards_counter = np.zeros(num_processes)
    time_step_counter = np.zeros(num_processes)
    episode_rewards = []
    time_steps = []
    last_save = time.time()
    solved_count = 0

    start = 0
    if load_path:
        state_dict = torch.load(load_path)
        if train_tasks:
            try:
                task_generator.load_state_dict(state_dict['task_generator'])
            except KeyError:
                pass
        actor_critic.load_state_dict(state_dict['actor_critic'])
        agent.optimizer.load_state_dict(state_dict['optimizer'])
        start = state_dict.get('step', -1) + 1
        if isinstance(envs.venv, VecNormalize):
            envs.venv.load_state_dict(state_dict['vec_normalize'])
        print(f'Loaded parameters from {load_path}.')

    if num_frames:
        end = start + int(num_frames) // num_steps // num_processes
        updates = range(start, end)
    else:
        updates = itertools.count(start)

    actor_critic.to(device)
    if train_tasks:
        task_generator.to(device)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)
    if train_tasks:
        tasks, probs = map(torch.tensor, envs.unwrapped.get_tasks_and_probs())
        importance_weights = task_generator.importance_weight(probs)
        rollouts.tasks[0].copy_(tasks.view(rollouts.tasks[0].size()))
        rollouts.importance_weighting[0].copy_(
            importance_weights.view(rollouts.importance_weighting[0].size()))

    start = time.time()
    for j in updates:

        if train_tasks:
            for i in range(num_processes):
                envs.unwrapped.set_task_dist(i, task_generator.probs())

        for step in range(num_steps):
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
                tasks, probs = map(torch.tensor,
                                   envs.unwrapped.get_tasks_and_probs())
                for i in tasks.numpy()[dones]:
                    last_n_tasks.append(i)
                    task_counts[i] += 1
                rollouts.insert(
                    obs=obs,
                    recurrent_hidden_states=recurrent_hidden_states,
                    actions=actions,
                    action_log_probs=action_log_probs,
                    values=values,
                    rewards=rewards,
                    masks=masks,
                    task=tasks,
                    importance_weighting=task_generator.importance_weight(
                        probs),
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

        train_results, *task_stuff = agent.update(rollouts, gamma)
        if train_tasks:
            tasks_trained, grads_per_task = task_stuff
            for k, v in grads_per_task.items():
                last_gradient[k] = v

        rollouts.after_update()
        total_num_steps = (j + 1) * num_processes * num_steps

        if log_dir and save_interval and \
                time.time() - last_save >= save_interval:
            last_save = time.time()
            modules = dict(
                optimizer=agent.optimizer,
                actor_critic=actor_critic)  # type: Dict[str, torch.nn.Module]

            if isinstance(envs.venv, VecNormalize):
                modules.update(vec_normalize=envs.venv)

            if train_tasks:
                modules.update(gan=task_generator)
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
                for k, v in train_results.items():
                    writer.add_scalar(k, v, total_num_steps)

                if train_tasks:

                    def plot(heatmap_values, name):
                        unwrapped = sample_env.unwrapped
                        fig = plt.figure()
                        if isinstance(unwrapped, GridWorld):
                            desc = np.zeros(unwrapped.desc.shape)
                            desc[unwrapped.decode(
                                unwrapped.task_states)] = heatmap_values
                            im = plt.imshow(desc, origin='lower')
                            plt.colorbar(im)
                        elif isinstance(unwrapped, AutoCurriculumHSREnv):
                            plt.bar(
                                np.arange(len(heatmap_values)), heatmap_values)
                        else:
                            return
                        writer.add_figure(name, fig, total_num_steps)
                        plt.close()

                    unique_values, unique_counts = np.unique(
                        last_n_tasks.array().astype(int), return_counts=True)

                    count_history = np.zeros(num_tasks)
                    count_history[unique_values] = unique_counts
                    plot(count_history, 'last 1e4 tasks')
                    plot(task_counts, 'all tasks')
                    plot(last_gradient.to('cpu').numpy(), 'last gradient')

            episode_rewards = []
            time_steps = []

        if eval_interval is not None and j % eval_interval == 0:
            eval_rewards = np.zeros(num_eval)
            eval_time_steps = np.zeros(num_eval)
            eval_dones = np.zeros(num_eval)

            obs = eval_envs.reset()
            eval_recurrent_hidden_states = torch.zeros(
                num_eval,
                actor_critic.recurrent_hidden_state_size,
                device=device)
            eval_masks = torch.zeros(num_eval, 1, device=device)

            while not np.all(eval_dones):
                with torch.no_grad():
                    _, actions, _, eval_recurrent_hidden_states = actor_critic.act(
                        inputs=obs,
                        rnn_hxs=eval_recurrent_hidden_states,
                        masks=eval_masks,
                        deterministic=deterministic_eval)

                # Observe reward and next obs
                obs, rewards, dones, infos = eval_envs.step(actions)

                # track rewards
                not_done = eval_dones == 0
                eval_rewards[not_done] += rewards.numpy()[not_done]
                eval_time_steps[not_done] += 1
                eval_dones[dones] = True

                eval_masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in dones])

            mean_returns = np.mean(eval_rewards)
            if log_dir:
                writer.add_scalar('eval return', mean_returns, total_num_steps)
                writer.add_scalar('eval time steps', np.mean(eval_time_steps),
                                  total_num_steps)

            print(" Evaluation using {} episodes: mean return {:.5f}\n".format(
                num_tasks, mean_returns))

            if solved is not None:
                if mean_returns >= solved:
                    solved_count += 1
                else:
                    solved_count = 0

            if solved_count == num_solved:
                print('Environment solved.')
                return
