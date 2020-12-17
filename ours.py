import multiprocessing
import pickle
import sys
from dataclasses import dataclass
from multiprocessing import Queue
from pathlib import Path
from queue import Empty, Full
from typing import Optional

import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

import data_types
import debug_env as _debug_env
import env
import osx_queue
import our_agent
import trainer
from aggregator import InfosAggregator
from common.vec_env import VecEnv
from config import BaseConfig
from data_types import CurriculumSetting
from utils import Discrete
from wrappers import VecPyTorch


@dataclass
class OurConfig(BaseConfig, env.EnvConfig):
    curriculum_level: int = 0
    curriculum_setting_load_path: Optional[str] = None
    curriculum_threshold: float = 0.9
    debug_env: bool = False
    failure_buffer_load_path: Optional[str] = None
    failure_buffer_size: int = 10000
    max_eval_lines: int = 50
    min_eval_lines: int = 1
    conv_hidden_size: int = 100
    debug: bool = False
    gate_coef: float = 0.01
    resources_hidden_size: int = 128
    kernel_size: int = 2
    lower_embed_size: int = 75
    max_curriculum_level: int = 10
    max_lines: int = 10
    min_lines: int = 1
    num_edges: int = 1
    no_pointer: bool = False
    no_roll: bool = False
    no_scan: bool = False
    olsk: bool = False
    stride: int = 1
    task_embed_size: int = 128
    transformer: bool = False


class Trainer(trainer.Trainer):
    @classmethod
    def args_to_methods(cls):
        mapping = super().args_to_methods()
        mapping["env_args"] += [
            env.Env.__init__,
            trainer.Trainer.make_vec_envs,
        ]
        mapping["agent_args"] += [our_agent.Agent.__init__]
        return mapping

    @staticmethod
    def build_agent(envs: VecPyTorch, **agent_args):
        return our_agent.Agent(
            observation_space=envs.observation_space,
            action_space=envs.action_space,
            **agent_args,
        )

    @staticmethod
    def build_failure_buffer(failure_buffer_load_path: Path, failure_buffer_size: int):
        failure_buffer = Queue(maxsize=failure_buffer_size)
        try:
            failure_buffer.qsize()
        except NotImplementedError:
            failure_buffer = osx_queue.Queue()
        if failure_buffer_load_path:
            with open(failure_buffer_load_path, "rb") as f:
                for x in pickle.load(f):
                    failure_buffer.put_nowait(x)

                print(
                    f"Loaded failure buffer of length {failure_buffer.qsize()} "
                    f"from {failure_buffer_load_path}"
                )
        return failure_buffer

    @classmethod
    def dump_failure_buffer(cls, failure_buffer: Queue, log_dir: Path):
        def gen():
            while True:
                try:
                    item = failure_buffer.get_nowait()
                except Empty:
                    return
                yield item
                try:
                    failure_buffer.put_nowait(item)
                except Full:
                    pass

        with Path(log_dir, "failure_buffer.pkl").open("wb") as f:
            buffer_list = [*gen()]
            pickle.dump(buffer_list, f)

    @classmethod
    def initialize_curriculum(
        cls,
        curriculum_level: int,
        curriculum_setting_load_path: Path,
        log_dir: Path,
        max_curriculum_level: int,
        max_lines: int,
        min_lines: int,
        max_eval_lines: int,
        min_eval_lines: int,
        curriculum_threshold: float,
    ):
        mean_successes = 0.5

        assert min_lines >= 1
        assert max_lines >= min_lines
        if curriculum_setting_load_path:
            with open(curriculum_setting_load_path, "rb") as f:
                curriculum_setting = pickle.load(f)
                print(
                    f"Loaded curriculum setting {curriculum_setting} "
                    f"from {curriculum_setting_load_path}"
                )
        else:
            curriculum_setting = CurriculumSetting(
                max_build_tree_depth=1000,
                max_lines=max_lines,
                n_lines_space=Discrete(min_lines, max_lines),
                level=0,
            )

        def curriculum_generator(setting: CurriculumSetting):
            while True:
                if setting.level == max_curriculum_level:
                    yield setting
                    continue
                if setting.n_lines_space.high < setting.max_lines:
                    setting = setting.increment_max_lines().increment_level()
                    yield setting
                setting = setting.increment_build_tree_depth().increment_level()
                yield setting

        curriculum_iterator = curriculum_generator(curriculum_setting)

        for _ in range(curriculum_level - curriculum_setting.level):
            curriculum_setting = next(curriculum_iterator)

        print(f"starting at curriculum: {curriculum_setting}")
        with Path(log_dir, "curriculum_setting.pkl").open("wb") as f:
            pickle.dump(curriculum_setting, f)

        while True:
            infos: InfosAggregator
            venv: VecEnv
            venv, infos = yield curriculum_setting

            try:
                mean_successes += (
                    0.1 * np.mean(infos.complete_episodes["success"])
                    - 0.9 * mean_successes
                )
            except KeyError:
                pass
            if mean_successes >= curriculum_threshold:
                mean_successes = 0.5
                curriculum_setting = next(curriculum_iterator)

                venv.set_curriculum(curriculum_setting)
                with Path(log_dir, "curriculum_setting.pkl").open("wb") as f:
                    pickle.dump(curriculum_setting, f)

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
        evaluating: bool,
        failure_buffer: Queue,
        world_size: int,
        **kwargs,
    ):
        data_types.WORLD_SIZE = world_size
        return super().make_vec_envs(
            evaluating=evaluating,
            non_pickle_args=dict(failure_buffer=failure_buffer),
            world_size=world_size,
            **kwargs,
        )


@hydra.main(config_name="config")
def app(cfg: DictConfig) -> None:
    Trainer.main(cfg)


if __name__ == "__main__":
    if sys.platform == "darwin":
        multiprocessing.set_start_method("fork")  # needed for osx_queue.Queue

    cs = ConfigStore.instance()
    cs.store(name="config", node=OurConfig)
    app()
