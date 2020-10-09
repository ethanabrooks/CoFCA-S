import inspect
from argparse import ArgumentParser
from pathlib import Path

import ours
import upper_agent
import upper_env
import main
from configs import default_upper
from trainer import Trainer


class UpperTrainer(Trainer):
    metric = "eval_reward"

    def build_agent(self, envs, debug=False, **agent_args):
        del agent_args["recurrent"]
        del agent_args["num_layers"]
        return upper_agent.Agent(
            observation_space=envs.observation_space,
            action_space=envs.action_space,
            debug=debug,
            **agent_args,
        )

    @staticmethod
    def make_env(seed, rank, evaluating, env_id=None, **kwargs):
        kwargs.update(
            seed=seed + rank,
            rank=rank,
            evaluating=evaluating,
        )
        return upper_env.Env(**kwargs)

    @classmethod
    def structure_config(cls, **config):
        config = super().structure_config(**config)
        agent_args = config.pop("agent_args")
        env_args = {}
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
        parser.add_argument("--lower-level-config", type=Path)
        parser.add_argument("--debug", action="store_true")
        parser.add_argument("--debug-obs", action="store_true")
        parser.add_argument("--no-scan", action="store_true")
        parser.add_argument("--no-roll", action="store_true")
        parser.add_argument("--no-pointer", action="store_true")
        parser.add_argument("--olsk", action="store_true")
        parser.add_argument("--transformer", action="store_true")
        parser.add_argument("--fuzz", action="store_true")
        parser.add_argument("--conv-hidden-size", type=int)
        parser.add_argument("--task-embed-size", type=int)
        parser.add_argument("--lower-embed-size", type=int)
        parser.add_argument("--inventory-hidden-size", type=int)
        parser.add_argument("--num-edges", type=int)
        parser.add_argument("--gate-coef", type=float)
        parser.add_argument("--no-op-coef", type=float)
        parser.add_argument("--kernel-size", type=int)
        parser.add_argument("--stride", type=int)

    @classmethod
    def add_arguments(cls, parser):
        parser = main.add_arguments(parser)
        parser.main.add_argument("--min-eval-lines", type=int)
        parser.main.add_argument("--max-eval-lines", type=int)
        parser.main.add_argument("--no-eval", action="store_true")
        parser.main.add_argument("--lower-level-load-path")
        env_parser = parser.main.add_argument_group("env_args")
        cls.add_env_arguments(env_parser)
        cls.add_agent_arguments(parser.agent)
        return parser.main

    @classmethod
    def launch(cls, env_id, config, **kwargs):
        if config is None:
            config = default_upper
        super().launch(env_id="experiment", config=config, **kwargs)

    @classmethod
    def main(cls):
        parser = ArgumentParser()
        cls.add_arguments(parser)
        cls.launch(**vars(parser.parse_args()))


if __name__ == "__main__":
    UpperTrainer.main()
