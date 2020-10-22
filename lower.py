import inspect

import lower_agent
import lower_env
import upper_env
from configs import default_lower
from trainer import Trainer
from upper import UpperTrainer


class LowerTrainer(UpperTrainer):
    metric = "reward"
    default = default_lower

    def build_agent(self, envs, **agent_args):
        return lower_agent.Agent(
            obs_spaces=envs.observation_space,
            action_space=envs.action_space,
            **agent_args,
        )

    @staticmethod
    def make_env(seed, rank, evaluating, env_id=None, **kwargs):
        kwargs.update(
            seed=seed + rank,
            rank=rank,
            evaluating=evaluating,
        )
        return lower_env.Env(**kwargs)

    @classmethod
    def structure_config(cls, **config):
        config = Trainer.structure_config(**config)
        agent_args = config.pop("agent_args")
        env_args = {}
        gen_args = {}

        for k, v in config.items():
            if (
                k in inspect.signature(upper_env.Env.__init__).parameters
                or k in inspect.signature(cls.make_env).parameters
            ):
                env_args[k] = v
            if (
                k in inspect.signature(lower_agent.Agent.__init__).parameters
                or k in inspect.signature(lower_agent.LowerLevel.__init__).parameters
            ):
                agent_args[k] = v
            if k in inspect.signature(cls.run).parameters:
                gen_args[k] = v

        config = dict(env_args=env_args, agent_args=agent_args, **gen_args)
        return config

    @classmethod
    def launch(cls, eval_interval, **kwargs):
        super().launch(eval_interval=None, **kwargs)

    @classmethod
    def add_agent_arguments(cls, parser):
        parser.add_argument("--num-conv-layers", type=int)
        parser.add_argument("--kernel-size", type=int)
        parser.add_argument("--stride", type=int)
        parser.add_argument("--sum-or-max")


if __name__ == "__main__":
    LowerTrainer.main()
