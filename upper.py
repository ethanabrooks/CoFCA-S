import inspect
import pickle
from itertools import islice
from pathlib import Path

import torch.nn as nn

import ours
import upper_agent
import debug_env as _debug_env
import networks
import upper_env
from aggregator import InfosAggregator
from configs import default_upper
from trainer import Trainer


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
    metric = "eval_reward"
    default = default_upper

    def build_infos_aggregator(self):
        return InfosAggregatorWithFailureBufferWriter()

    def report_generator(self, log_dir):
        reporter = super().report_generator(log_dir)
        next(reporter)

        def report(failure_buffer, **kwargs):
            with Path(log_dir, "failure_buffer.pkl").open("wb") as f:
                pickle.dump(failure_buffer, f)
            reporter.send(kwargs)

        while True:
            msg = yield
            report(**msg)

    def build_agent(self, envs, debug=False, hard_code_lower=False, **agent_args):
        del agent_args["recurrent"]
        del agent_args["num_layers"]
        return upper_agent.Agent(
            observation_space=envs.observation_space,
            action_space=envs.action_space,
            debug=debug,
            envs=envs if hard_code_lower else None,
            **agent_args,
        )

    @staticmethod
    def make_env(seed, rank, evaluating, debug_env=False, env_id=None, **kwargs):
        kwargs.update(
            seed=seed + rank,
            rank=rank,
            evaluating=evaluating,
        )
        if debug_env:
            return _debug_env.Env(**kwargs)
        else:
            return upper_env.Env(**kwargs)

    @classmethod
    def structure_config(cls, **config):
        config = super().structure_config(**config)
        agent_args = config.pop("agent_args")
        env_args = config.pop("env_args")
        gen_args = {}

        # agent_args["eval_lines"] = config["max_eval_lines"]
        # agent_args["debug"] = config["render"] or config["render_eval"]

        for k, v in config.items():
            if (
                k in inspect.signature(upper_env.Env.__init__).parameters
                or k in inspect.signature(cls.make_env).parameters
            ):
                if k == "lower_level":
                    if v:
                        print("lower_level specified. Using gridworld env")
                    else:
                        print("lower_level not specified. Using debug_env")
                env_args[k] = v
            if k in inspect.signature(ours.Recurrence.__init__).parameters:
                agent_args[k] = v
            if k in inspect.signature(upper_agent.Agent.__init__).parameters:
                agent_args[k] = v
            if k in inspect.signature(upper_agent.Agent.__init__).parameters:
                agent_args[k] = v
            if k in inspect.signature(cls.run).parameters:
                gen_args[k] = v
        return dict(env_args=env_args, agent_args=agent_args, **gen_args)

    @classmethod
    def add_env_arguments(cls, parser):
        upper_env.Env.add_arguments(parser)

    @classmethod
    def add_agent_arguments(cls, parser):
        parser.add_argument("--conv-hidden-size", type=int)
        parser.add_argument("--debug", action="store_true")
        parser.add_argument("--debug-obs", action="store_true")
        parser.add_argument("--fuzz", action="store_true")
        parser.add_argument("--gate-coef", type=float)
        parser.add_argument("--hard-code-lower", action="store_true")
        parser.add_argument("--inventory-hidden-size", type=int)
        parser.add_argument("--kernel-size", type=int)
        parser.add_argument("--lower-embed-size", type=int)
        parser.add_argument("--lower-level-config", type=Path)
        parser.add_argument("--olsk", action="store_true")
        parser.add_argument("--num-edges", type=int)
        parser.add_argument("--no-op-coef", type=float)
        parser.add_argument("--no-pointer", action="store_true")
        parser.add_argument("--no-roll", action="store_true")
        parser.add_argument("--no-scan", action="store_true")
        parser.add_argument("--stride", type=int)
        parser.add_argument("--task-embed-size", type=int)
        parser.add_argument("--transformer", action="store_true")

    @classmethod
    def add_arguments(cls, parser):
        parser = super().add_arguments(parser)
        parser.main.add_argument("--min-eval-lines", type=int)
        parser.main.add_argument("--max-eval-lines", type=int)
        parser.main.add_argument("--no-eval", action="store_true")
        parser.main.add_argument("--lower-level-load-path")
        env_parser = parser.main.add_argument_group("env_args")
        cls.add_env_arguments(env_parser)
        cls.add_agent_arguments(parser.agent)
        return parser

    @classmethod
    def launch(cls, env_id, **kwargs):
        super().launch(env_id="experiment", **kwargs)


if __name__ == "__main__":
    UpperTrainer.main()
