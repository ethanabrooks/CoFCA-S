import pickle
from pathlib import Path
from multiprocessing import Queue
from typing import Optional

import numpy as np

import baseline_agent
import debug_env as _debug_env
import env
import our_agent
import our_recurrence
import trainer
from aggregator import InfosAggregator
import osx_queue
from common.vec_env import VecEnv, VecEnvWrapper
from data_types import CurriculumSetting
from utils import Discrete
from wrappers import VecPyTorch


class CurriculumWrapper(VecEnvWrapper):
    def __init__(
        self,
        venv: VecEnv,
        curriculum_setting: CurriculumSetting,
        curriculum_threshold: float,
        log_dir: Path,
    ):
        super().__init__(venv)
        self.log_dir = log_dir
        self.curriculum_threshold = curriculum_threshold
        self.curriculum_iterator = self.curriculum_generator(curriculum_setting)
        next(self.curriculum_iterator)

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        return self.venv.step_wait()

    def preprocess(self, action):
        return self.venv.preprocess(action)

    def to(self, device):
        return self.venv.to(device)

    @staticmethod
    def curriculum_generator(setting: CurriculumSetting):
        while True:
            if setting.n_lines_space.high < setting.max_lines:
                setting = setting.increment_max_lines().increment_level()
                yield setting
            setting = setting.increment_build_tree_depth().increment_level()
            yield setting

    def process_infos(self, infos: InfosAggregator):
        mean_success = np.mean(infos.complete_episodes.get("success", [0]))
        if mean_success >= self.curriculum_threshold:
            curriculum = next(self.curriculum_iterator)
            self.venv.set_curriculum(curriculum)
            with Path(self.log_dir, "curriculum_setting.pkl").open("wb") as f:
                pickle.dump(curriculum, f)


class Trainer(trainer.Trainer):
    @classmethod
    def add_arguments(cls, parser):
        parser = super().add_arguments(parser)
        parser.main.add_argument("--curriculum_threshold", type=float, default=0.9)
        parser.main.add_argument("--curriculum_setting_load_path", type=Path)
        parser.main.add_argument("--eval", dest="no_eval", action="store_false")
        parser.main.add_argument("--failure_buffer_load_path", type=Path)
        parser.main.add_argument("--failure_buffer_size", type=int, default=10000)
        parser.main.add_argument("--min_eval_lines", type=int, default=1)
        parser.main.add_argument("--max_eval_lines", type=int, default=50)
        env_parser = parser.main.add_argument_group("env_args")
        env.Env.add_arguments(env_parser)
        parser.agent.add_argument("--conv_hidden_size", type=int, default=100)
        parser.agent.add_argument("--debug", action="store_true")
        parser.agent.add_argument("--gate_coef", type=float, default=0.01)
        parser.agent.add_argument("--resources_hidden_size", type=int, default=128)
        parser.agent.add_argument("--kernel_size", type=int, default=2)
        parser.agent.add_argument("--lower_embed_size", type=int, default=75)
        parser.agent.add_argument("--max_lines", type=int, default=10)
        parser.agent.add_argument("--min_lines", type=int, default=1)
        parser.agent.add_argument("--next_actions_embed_size", type=int, default=25)
        parser.agent.add_argument("--num_edges", type=int, default=1)
        parser.agent.add_argument("--no_pointer", action="store_true")
        parser.agent.add_argument("--no_roll", action="store_true")
        parser.agent.add_argument("--no_scan", action="store_true")
        parser.agent.add_argument("--olsk", action="store_true")
        parser.agent.add_argument("--stride", type=int, default=1)
        parser.agent.add_argument("--task_embed_size", type=int, default=128)
        parser.agent.add_argument("--transformer", action="store_true")
        return parser

    @classmethod
    def args_to_methods(cls):
        mapping = super().args_to_methods()
        mapping["env_args"] += [
            env.Env.__init__,
            CurriculumWrapper.__init__,
            trainer.Trainer.make_vec_envs,
        ]
        mapping["agent_args"] += [
            our_recurrence.Recurrence.__init__,
            our_agent.Agent.__init__,
        ]
        return mapping

    @staticmethod
    def build_agent(envs: VecPyTorch, **agent_args):
        del agent_args["recurrent"]
        del agent_args["num_layers"]
        return baseline_agent.Agent(
            observation_space=envs.observation_space,
            action_space=envs.action_space,
            **agent_args,
        )

    @staticmethod
    def make_env(
        rank: int,
        seed: int,
        debug_env=False,
        env_id=None,
        **kwargs,
    ):
        kwargs.update(rank=rank, random_seed=seed + rank)
        if debug_env:
            return _debug_env.Env(**kwargs)
        else:
            return env.Env(**kwargs)

    # noinspection PyMethodOverriding
    @classmethod
    def make_vec_envs(
        cls,
        curriculum_setting_load_path: Optional[Path],
        curriculum_threshold: float,
        evaluating: bool,
        failure_buffer_load_path: Path,
        failure_buffer_size: int,
        log_dir: Path,
        max_eval_lines: int,
        max_lines: int,
        min_eval_lines: int,
        min_lines: int,
        **kwargs,
    ):
        assert min_lines >= 1
        assert max_lines >= min_lines
        if curriculum_setting_load_path:
            with curriculum_setting_load_path.open("rb") as f:
                curriculum_setting = pickle.load(f)
                print(
                    f"Loaded curriculum setting {curriculum_setting} "
                    f"from {curriculum_setting_load_path}"
                )
        elif evaluating:
            curriculum_setting = CurriculumSetting(
                max_build_tree_depth=100,
                max_lines=max_eval_lines,
                n_lines_space=Discrete(min_eval_lines, min_eval_lines),
                level=0,
            )
        else:
            curriculum_setting = CurriculumSetting(
                max_build_tree_depth=1,
                max_lines=max_lines,
                n_lines_space=Discrete(min_lines, min_lines),
                level=0,
            )
        kwargs.update(
            curriculum_setting=curriculum_setting,
        )

        if failure_buffer_load_path:
            with failure_buffer_load_path.open("rb") as f:
                failure_buffer = pickle.load(f)
                assert isinstance(failure_buffer, Queue)
                print(
                    f"Loaded failure buffer of length {failure_buffer.qsize()} "
                    f"from {failure_buffer_load_path}"
                )
        else:
            failure_buffer = Queue(maxsize=failure_buffer_size)
            try:
                failure_buffer.qsize()
            except NotImplementedError:
                failure_buffer = osx_queue.Queue()
        venv = super().make_vec_envs(
            evaluating=evaluating,
            non_pickle_args=dict(failure_buffer=failure_buffer),
            **kwargs,
        )
        return CurriculumWrapper(
            venv=venv,
            curriculum_setting=curriculum_setting,
            curriculum_threshold=curriculum_threshold,
            log_dir=log_dir,
        )


if __name__ == "__main__":
    Trainer.main()
