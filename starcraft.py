import pickle
from itertools import islice
from pathlib import Path

import debug_env as _debug_env
import env
import our_agent
import ours
import trainer
from aggregator import InfosAggregator
from common.vec_env import VecEnv
from wrappers import VecPyTorch
import numpy as np


class InfosAggregatorWithFailureBufferWriter(InfosAggregator):
    def __init__(self):
        super().__init__()
        self.failure_buffers = {}

    def update(self, *infos: dict, dones):
        for i, info in enumerate(infos):
            try:
                self.failure_buffers[i] = info.pop("failure_buffer")
            except KeyError:
                pass

        super().update(*infos, dones=dones)

    def concat_buffers(self):
        for buffer in self.failure_buffers.values():
            yield from buffer

    def items(self):
        failure_buffer = list(islice(self.concat_buffers(), 50))
        yield "failure_buffer", failure_buffer
        yield from super().items()

    def mean_success(self):
        return np.mean(self.complete_episodes.get("success", [0]))


class Trainer(trainer.Trainer):
    metric = "reward"

    @classmethod
    def add_agent_arguments(cls, parser):
        parser.add_argument("--conv_hidden_size", type=int, default=100)
        parser.add_argument("--debug", action="store_true")
        parser.add_argument("--gate_coef", type=float, default=0.01)
        parser.add_argument("--resources_hidden_size", type=int, default=128)
        parser.add_argument("--kernel_size", type=int, default=2)
        parser.add_argument("--lower_embed_size", type=int, default=75)
        parser.add_argument("--olsk", action="store_true")
        parser.add_argument("--next_actions_embed_size", type=int, default=25)
        parser.add_argument("--num_edges", type=int, default=1)
        parser.add_argument("--no_pointer", action="store_true")
        parser.add_argument("--no_roll", action="store_true")
        parser.add_argument("--no_scan", action="store_true")
        parser.add_argument("--stride", type=int, default=1)
        parser.add_argument("--task_embed_size", type=int, default=128)
        parser.add_argument("--transformer", action="store_true")

    @classmethod
    def add_arguments(cls, parser):
        parser = super().add_arguments(parser)
        parser.main.add_argument("--curriculum_threshold", type=float, default=0.9)
        parser.main.add_argument("--eval", dest="no_eval", action="store_false")
        parser.main.add_argument("--min_eval_lines", type=int, default=1)
        parser.main.add_argument("--max_eval_lines", type=int, default=50)
        env_parser = parser.main.add_argument_group("env_args")
        cls.add_env_arguments(env_parser)
        cls.add_agent_arguments(parser.agent)
        return parser

    @classmethod
    def add_env_arguments(cls, parser):
        env.Env.add_arguments(parser)

    @classmethod
    def args_to_methods(cls):
        mapping = super().args_to_methods()
        mapping["env_args"] += [env.Env.__init__]
        mapping["agent_args"] += [ours.Recurrence.__init__, our_agent.Agent.__init__]
        return mapping

    @staticmethod
    def build_agent(envs: VecPyTorch, debug=False, **agent_args):
        del agent_args["recurrent"]
        del agent_args["num_layers"]
        return our_agent.Agent(
            observation_space=envs.observation_space,
            action_space=envs.action_space,
            debug=debug,
            **agent_args,
        )

    @staticmethod
    def build_infos_aggregator():
        return InfosAggregatorWithFailureBufferWriter()

    # noinspection PyMethodOverriding
    @staticmethod
    def handle_curriculum(
        infos: InfosAggregatorWithFailureBufferWriter,
        envs: VecEnv,
        curriculum_threshold: float,
    ):
        if infos.mean_success() >= curriculum_threshold:
            envs.increment_curriculum()

    @staticmethod
    def make_env(
        seed,
        rank,
        evaluating,
        min_lines=None,
        max_lines=None,
        min_eval_lines=None,
        max_eval_lines=None,
        debug_env=False,
        env_id=None,
        **kwargs
    ):
        if evaluating:
            min_lines = min_eval_lines
            max_lines = max_eval_lines
        kwargs.update(
            evaluating=evaluating,
            min_lines=min_lines,
            max_lines=max_lines,
            rank=rank,
            random_seed=seed + rank,
        )
        if debug_env:
            return _debug_env.Env(**kwargs)
        else:
            return env.Env(**kwargs)

    @classmethod
    def report(cls, failure_buffer, log_dir: Path, **kwargs):
        if failure_buffer is not None:
            with Path(log_dir, "failure_buffer.pkl").open("wb") as f:
                pickle.dump(failure_buffer, f)
        super().report(**kwargs, log_dir=log_dir)


if __name__ == "__main__":
    Trainer.main()
