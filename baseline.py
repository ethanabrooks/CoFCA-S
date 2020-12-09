from queue import Queue

import baseline_agent
import our_agent
import env
import ours
import trainer
from common.vec_env.dummy_vec_env import DummyVecEnv
from common.vec_env.subproc_vec_env import SubprocVecEnv
from data_types import CurriculumSetting
from utils import Discrete
from wrappers import VecPyTorch


class Trainer(ours.Trainer):
    @staticmethod
    def build_agent(envs: VecPyTorch, **agent_args):
        del agent_args["recurrent"]
        del agent_args["num_layers"]
        return baseline_agent.Agent(
            observation_space=envs.observation_space,
            action_space=envs.action_space,
            **agent_args,
        )

    @classmethod
    def initial_curriculum(cls, min_lines, max_lines):
        return CurriculumSetting(
            max_build_tree_depth=100,
            max_lines=max_lines,
            n_lines_space=Discrete(min_lines, max_lines),
            level=0,
        )


if __name__ == "__main__":
    Trainer.main()
