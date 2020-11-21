import pickle
from itertools import islice
from pathlib import Path

import debug_env as _debug_env
import env
import our_agent
import ours
from aggregator import InfosAggregator
from configs import starcraft_default
from trainer import Trainer
from wrappers import VecPyTorch


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


class UpperTrainer(Trainer):
    metric = "reward"
    default = starcraft_default

    def build_infos_aggregator(self):
        return InfosAggregatorWithFailureBufferWriter()

    def report_generator(self, log_dir):
        reporter = super().report_generator(log_dir)
        next(reporter)

        def report(failure_buffer=None, **kwargs):
            if failure_buffer is not None:
                with Path(log_dir, "failure_buffer.pkl").open("wb") as f:
                    pickle.dump(failure_buffer, f)
            reporter.send(kwargs)

        while True:
            msg = yield
            report(**msg)

    def build_agent(self, envs: VecPyTorch, debug=False, **agent_args):
        del agent_args["recurrent"]
        del agent_args["num_layers"]
        return our_agent.Agent(
            observation_space=envs.observation_space,
            action_space=envs.action_space,
            debug=debug,
            **agent_args,
        )

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
        # if evaluating:
        #     min_lines = min_eval_lines
        #     max_lines = max_eval_lines
        kwargs.update(
            evaluating=evaluating,
            min_lines=min_lines,
            max_lines=max_lines,
            min_eval_lines=min_eval_lines,
            max_eval_lines=max_eval_lines,
            rank=rank,
            random_seed=seed + rank,
        )
        if debug_env:
            return _debug_env.Env(**kwargs)
        else:
            return env.Env(**kwargs)

    @classmethod
    def args_to_methods(cls):
        mapping = super().args_to_methods()
        mapping["env_args"] += [env.Env.__init__]
        mapping["agent_args"] += [ours.Recurrence.__init__, our_agent.Agent.__init__]
        return mapping

    @classmethod
    def add_env_arguments(cls, parser):
        env.Env.add_arguments(parser)

    @classmethod
    def add_agent_arguments(cls, parser):
        parser.add_argument("--debug", action="store_true")
        parser.add_argument("--no-scan", action="store_true")
        parser.add_argument("--no-roll", action="store_true")
        parser.add_argument("--no-pointer", action="store_true")
        parser.add_argument("--olsk", action="store_true")
        parser.add_argument("--transformer", action="store_true")
        parser.add_argument("--conv-hidden-size", type=int)
        parser.add_argument("--task-embed-size", type=int)
        parser.add_argument("--lower-embed-size", type=int)
        parser.add_argument("--resources-hidden-size", type=int)
        parser.add_argument("--num-edges", type=int)
        parser.add_argument("--gate-coef", type=float)
        parser.add_argument("--no-op-coef", type=float)
        parser.add_argument("--kernel-size", type=int)
        parser.add_argument("--stride", type=int)

    @classmethod
    def add_arguments(cls, parser):
        parser = super().add_arguments(parser)
        parser.main.add_argument("--min-eval-lines", type=int)
        parser.main.add_argument("--max-eval-lines", type=int)
        parser.main.add_argument("--no-eval", action="store_true")
        env_parser = parser.main.add_argument_group("env_args")
        cls.add_env_arguments(env_parser)
        cls.add_agent_arguments(parser.agent)
        return parser

    @classmethod
    def launch(cls, env_id, **kwargs):
        super().launch(env_id="experiment", **kwargs)


if __name__ == "__main__":
    UpperTrainer.main()
