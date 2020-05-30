import itertools
import re
import time
from pathlib import Path

import torch
from rl_utils import hierarchical_parse_args
from tensorboardX import SummaryWriter

from ppo import subtasks
from ppo.arguments import build_parser
from ppo.train import Train
from ppo.utils import get_random_gpu, get_n_gpu, k_scalar_pairs
from ppo.water_maze import WaterMaze


def main(log_dir, seed, **kwargs):
    class _Train(Train):
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

        @staticmethod
        def make_env(seed, rank, evaluation, env_id, add_timestep, **env_args):
            return WaterMaze(**env_args, seed=seed + rank)

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


def args():
    parsers = build_parser()
    parser = parsers.main
    parser.add_argument("--no-tqdm", dest="use_tqdm", action="store_false")
    parser.add_argument("--time-limit", type=int)
    WaterMaze.add_arguments(parsers.env)
    return parser


if __name__ == "__main__":
    main(**hierarchical_parse_args(args()))
