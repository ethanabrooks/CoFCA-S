import itertools
import re
import time
from abc import ABC
from pathlib import Path

import ray
import torch
from ray.tune.result import TIME_TOTAL_S
from ray.tune.schedulers import AsyncHyperBandScheduler
from rl_utils import hierarchical_parse_args
from tensorboardX import SummaryWriter
import socket

from ppo import subtasks
from ppo.arguments import build_parser
from ppo.subtasks.agent import Agent
from ppo.train import Train
from ppo.utils import get_random_gpu, get_n_gpu, k_scalar_pairs


class TrainControlFlow(Train, ABC):
    @staticmethod
    def make_env(time_limit, seed, rank, evaluation, env_id, add_timestep, **env_args):
        return subtasks.bandit.Env(**env_args, seed=seed + rank)

    def build_agent(self, envs, **agent_args):
        return Agent(
            observation_space=envs.observation_space,
            action_space=envs.action_space,
            **agent_args,
        )


def tune_main(debug, redis_port, log_dir, num_samples, tune_metric, **kwargs):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    ray.init(redis_address=f"{ip}:{redis_port}", local_mode=debug)
    kwargs.update(
        use_gae=ray.tune.choice([True, False]),
        num_batch=ray.tune.choice([1]),
        num_steps=ray.tune.choice([16, 32, 64, 128]),
        seed=ray.tune.choice(list(range(10))),
        entropy_coef=ray.tune.uniform(low=0.01, high=0.04),
        hidden_size=ray.tune.choice([32, 64, 128, 256]),
        num_layers=ray.tune.choice([0, 1, 2]),
        learning_rate=ray.tune.uniform(low=0.0001, high=0.005),
        ppo_epoch=ray.tune.choice(list(range(1, 6))),
        # feed_r_initially=ray.tune.choice([True, False]),
        # use_M_plus_minus=ray.tune.choice([True, False]),
        num_processes=300,
        # ppo_epoch=ray.tune.sample_from(
        #     lambda spec: max(
        #         1,
        #         int(
        #             1.84240459e06 * spec.config.learning_rate ** 2
        #             - 1.14376715e04 * spec.config.learning_rate
        #             + 1.89339209e01
        #             + np.random.uniform(low=-2, high=2)
        #         ),
        #     )
        # ),
    )

    class _Train(TrainControlFlow, ray.tune.Trainable):
        def _setup(self, config):
            def setup(
                agent_args,
                ppo_args,
                entropy_coef,
                hidden_size,
                num_layers,
                learning_rate,
                ppo_epoch,
                **kwargs,
            ):
                agent_args.update(
                    entropy_coef=entropy_coef,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                )
                ppo_args.update(ppo_epoch=ppo_epoch, learning_rate=learning_rate)
                self.setup(**kwargs, agent_args=agent_args, ppo_args=ppo_args)

            setup(**config)

        def get_device(self):
            return "cuda"

    ray.tune.run(
        _Train,
        config=kwargs,
        resources_per_trial=dict(cpu=1, gpu=0.5 if torch.cuda.is_available() else 0),
        checkpoint_freq=1,
        reuse_actors=True,
        num_samples=1 if debug else num_samples,
        local_dir=log_dir,
        scheduler=AsyncHyperBandScheduler(
            time_attr=TIME_TOTAL_S,
            metric=tune_metric,
            mode="max",
            grace_period=3600,
            max_t=43200,
        ),
    )


def bandit_main(log_dir, seed, **kwargs):
    class _Train(TrainControlFlow):
        def __init__(
            self,
            run_id,
            log_dir: Path,
            save_interval: int,
            num_processes: int,
            num_steps: int,
            **kwargs,
        ):
            self.num_steps = num_steps
            self.num_processes = num_processes
            self.run_id = run_id
            self.save_interval = save_interval
            self.log_dir = log_dir
            if log_dir:
                self.writer = SummaryWriter(logdir=str(log_dir))
            else:
                self.writer = None
            self.setup(**kwargs, num_processes=num_processes, num_steps=num_steps)
            self.last_save = time.time()  # dummy save

        def run(self):
            for _ in itertools.count():
                for result in self.make_train_iterator():
                    if self.writer is not None:
                        total_num_steps = (
                            (self.i + 1) * self.num_processes * self.num_steps
                        )
                        for k, v in k_scalar_pairs(**result):
                            self.writer.add_scalar(k, v, total_num_steps)

                    if (
                        self.log_dir
                        and self.save_interval
                        and (time.time() - self.last_save >= self.save_interval)
                    ):
                        self._save(str(self.log_dir))
                        self.last_save = time.time()

        def get_device(self):
            match = re.search("\d+$", self.run_id)
            if match:
                device_num = int(match.group()) % get_n_gpu()
            else:
                device_num = get_random_gpu()

            return torch.device("cuda", device_num)

    _Train(**kwargs, seed=seed, log_dir=log_dir).run()


def bandit_args():
    parsers = build_parser()
    parser = parsers.main
    parser.add_argument("--no-tqdm", dest="use_tqdm", action="store_false")
    parser.add_argument("--time-limit", type=int)
    parsers.env.add_argument("--n-lines", type=int, required=True)
    parsers.env.add_argument("--flip-prob", type=float, required=True)
    parsers.agent.add_argument("--debug", action="store_true")
    parsers.agent.add_argument("--baseline", action="store_true")
    return parser


if __name__ == "__main__":
    bandit_main(**hierarchical_parse_args(bandit_args()))
