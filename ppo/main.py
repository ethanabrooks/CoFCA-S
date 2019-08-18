import itertools
import re
import time
from abc import ABC

import numpy as np

from pathlib import Path

import ray
from gym.wrappers import TimeLimit
from ray.tune.result import TIME_TOTAL_S
from ray.tune.schedulers import AsyncHyperBandScheduler
from rl_utils import hierarchical_parse_args
from tensorboardX import SummaryWriter

import ppo
import ppo.events.agent
from ppo.arguments import get_args, build_parser
from ppo.events.agent import Agent
from ppo.train import Train
from ppo.utils import get_random_gpu, get_n_gpu, k_scalar_pairs
import torch


def cli():
    Train(**get_args())


def exp_main(
    gridworld_args,
    wrapper_args,
    single_instruction,
    debug,
    tune,
    redis_address,
    log_dir,
    num_samples,
    baseline,
    **kwargs,
):
    class TrainEvents(Train, ABC):
        @staticmethod
        def make_env(time_limit, seed, rank, evaluation, env_id, add_timestep):
            env = ppo.events.Gridworld(**gridworld_args, seed=seed)
            if single_instruction:
                env = ppo.events.SingleInstructionWrapper(
                    **wrapper_args, evaluation=evaluation, env=env
                )
            else:
                env = ppo.events.Wrapper(**wrapper_args, evaluation=evaluation, env=env)
            env = TimeLimit(max_episode_steps=time_limit, env=env)
            env.seed(seed + rank)
            return env

        def build_agent(
            self,
            envs,
            hidden_size=None,
            num_layers=None,
            activation=None,
            entropy_coef=None,
            recurrent=None,
            feed_r_initially=None,
            use_M_plus_minus=None,
            device=None,
        ):
            agent_args = dict(
                hidden_size=hidden_size,
                num_layers=num_layers,
                activation=activation,
                entropy_coef=entropy_coef,
                recurrent=recurrent,
            )
            if single_instruction:
                return super().build_agent(envs, **agent_args)
            return Agent(
                observation_space=envs.observation_space,
                action_space=envs.action_space,
                debug=False if tune else debug,
                baseline=baseline,
                use_M_plus_minus=use_M_plus_minus,
                feed_r_initially=feed_r_initially,
                **agent_args,
            )

    if tune:
        ray.init(redis_address=redis_address, local_mode=debug)
        kwargs.update(
            use_gae=ray.tune.choice([True, False]),
            num_batch=ray.tune.choice([1, 2]),
            num_steps=ray.tune.choice([16, 32, 64]),
            seed=ray.tune.choice(list(range(10))),
            entropy_coef=ray.tune.uniform(low=0.01, high=0.04),
            hidden_size=ray.tune.choice([32, 64, 128, 512]),
            num_layers=ray.tune.choice([0, 1, 2]),
            learning_rate=ray.tune.uniform(low=0.0001, high=0.005),
            ppo_epoch=ray.tune.choice(list(range(5))),
            feed_r_initially=ray.tune.choice([True, False]),
            use_M_plus_minus=ray.tune.choice([True, False])
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

        class _Train(TrainEvents, ray.tune.Trainable):
            def _setup(self, config):
                def setup(
                    run_id,
                    save_interval,
                    agent_args,
                    ppo_args,
                    entropy_coef,
                    hidden_size,
                    num_layers,
                    learning_rate,
                    ppo_epoch,
                    feed_r_initially,
                    use_M_plus_minus,
                    **kwargs,
                ):
                    agent_args.update(
                        entropy_coef=entropy_coef,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        feed_r_initially=feed_r_initially,
                        use_M_plus_minus=use_M_plus_minus,
                    )
                    ppo_args.update(ppo_epoch=ppo_epoch, learning_rate=learning_rate)
                    self.setup(**kwargs, agent_args=agent_args, ppo_args=ppo_args)

                setup(**config)

            def get_device(self):
                return "cuda"

        ray.tune.run(
            _Train,
            config=kwargs,
            resources_per_trial=dict(
                cpu=1, gpu=0.5 if torch.cuda.is_available() else 0
            ),
            checkpoint_freq=1,
            reuse_actors=True,
            num_samples=1 if debug else num_samples,
            local_dir=log_dir,
            scheduler=AsyncHyperBandScheduler(
                time_attr=TIME_TOTAL_S,
                metric="eval_rewards",
                mode="max",
                grace_period=3600,
                max_t=43200,
            ),
        )
    else:

        class _Train(TrainEvents):
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

        _Train(**kwargs, log_dir=log_dir).run()


def exp_cli():
    parsers = build_parser()
    parser = parsers.main
    parser.add_argument("--single-instruction", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--redis-address")
    parsers.agent.add_argument("--feed-r-initially", action="store_true")
    parsers.agent.add_argument("--use-M-plus-minus", action="store_true")
    gridworld_parser = parser.add_argument_group("gridworld_args")
    gridworld_parser.add_argument("--height", help="", type=int, default=4)
    gridworld_parser.add_argument("--width", help="", type=int, default=4)
    gridworld_parser.add_argument("--cook-time", help="", type=int, default=2)
    gridworld_parser.add_argument("--time-to-heat-oven", help="", type=int, default=3)
    gridworld_parser.add_argument("--doorbell-prob", help="", type=float, default=0.05)
    gridworld_parser.add_argument("--mouse-prob", help="", type=float, default=0.2)
    gridworld_parser.add_argument("--baby-prob", help="", type=float, default=0.1)
    gridworld_parser.add_argument("--mess-prob", help="", type=float, default=0.01)
    gridworld_parser.add_argument("--fly-prob", help="", type=float, default=0.005)
    gridworld_parser.add_argument("--toward-cat-prob", help="", type=float, default=0.5)
    wrapper_parser = parser.add_argument_group("wrapper_args")
    wrapper_parser.add_argument(
        "--n-active-instructions", help="", type=int, required=True
    )
    wrapper_parser.add_argument("--vision-range", help="", type=float, default=1)
    wrapper_parser.add_argument("--watch-baby-range", help="", type=int, default=2)
    wrapper_parser.add_argument("--avoid-dog-range", help="", type=int, default=2)
    wrapper_parser.add_argument("--door-time-limit", help="", type=int, default=7)
    wrapper_parser.add_argument("--max-time-outside", help="", type=int, default=15)
    wrapper_parser.add_argument("--instruction", dest="instructions", action="append")
    wrapper_parser.add_argument("--test", nargs="*", action="append", default=[])
    wrapper_parser.add_argument("--valid", nargs="*", action="append", default=[])
    exp_main(**hierarchical_parse_args(parser))


if __name__ == "__main__":
    exp_cli()
