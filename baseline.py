from queue import Queue

import baseline_agent
import env
import ours
import trainer
from common.vec_env.dummy_vec_env import DummyVecEnv
from common.vec_env.subproc_vec_env import SubprocVecEnv
from data_types import CurriculumSetting
from utils import Discrete
from wrappers import VecPyTorch


class Trainer(ours.Trainer):
    @classmethod
    def add_arguments(cls, parser):
        parser = trainer.Trainer.add_arguments(parser)
        env_parser = parser.main.add_argument_group("env_args")
        env.Env.add_arguments(env_parser)
        parser.agent.add_argument("--conv_hidden_size", type=int, default=100)
        parser.agent.add_argument("--resources_hidden_size", type=int, default=128)
        parser.agent.add_argument("--kernel_size", type=int, default=2)
        parser.agent.add_argument("--lower_embed_size", type=int, default=75)
        parser.agent.add_argument("--next_actions_embed_size", type=int, default=25)
        parser.agent.add_argument("--stride", type=int, default=1)
        parser.agent.add_argument("--task_embed_size", type=int, default=128)
        return parser

    @classmethod
    def args_to_methods(cls):
        mapping = super().args_to_methods()
        mapping["agent_args"] += [baseline_agent.Agent.__init__]
        mapping["env_args"] = [env.Env.__init__, cls.make_vec_envs, cls.make_env]
        mapping["run_args"] = [trainer.Trainer.run]
        return mapping

    @staticmethod
    def build_agent(envs: VecPyTorch, **agent_args):
        del agent_args["recurrent"]
        del agent_args["num_layers"]
        return baseline_agent.Agent(
            observation_space=envs.observation_space,
            action_space=envs.action_space,
            debug=False,
            max_eval_lines=1,
            no_pointer=False,
            no_roll=False,
            no_scan=False,
            num_edges=1,
            olsk=False,
            transformer=False,
            **agent_args,
        )

    # noinspection PyMethodOverriding
    @classmethod
    def make_vec_envs(
        cls,
        evaluating: bool,
        num_processes: int,
        render: bool,
        synchronous: bool,
        **kwargs,
    ) -> VecPyTorch:
        kwargs.update(
            eval_steps=0,
            failure_buffer=Queue(),
            tgt_success_rate=1,
            curriculum_setting=CurriculumSetting(
                max_build_tree_depth=100,
                max_lines=1,
                n_lines_space=Discrete(1, 1),
                level=0,
            ),
        )

        def env_thunk(rank):
            return lambda: cls.make_env(rank=rank, evaluating=evaluating, **kwargs)

        env_fns = [env_thunk(i) for i in range(num_processes)]
        return VecPyTorch(
            DummyVecEnv(env_fns, render=render)
            if len(env_fns) == 1 or synchronous
            else SubprocVecEnv(env_fns)
        )

    @classmethod
    def run(cls, *args, **kwargs):
        super().run(*args, **kwargs, no_eval=True)


if __name__ == "__main__":
    Trainer.main()
