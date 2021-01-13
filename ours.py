import pickle
import sys
from dataclasses import dataclass
from multiprocessing import Queue
from pathlib import Path
from pprint import pprint
from queue import Empty, Full
from typing import Optional, Dict

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

import data_types
import env
import osx_queue
import our_agent
import trainer
from config import BaseConfig
from wrappers import VecPyTorch


@dataclass
class OurConfig(BaseConfig, env.EnvConfig, our_agent.AgentConfig):
    failure_buffer_load_path: Optional[str] = None
    failure_buffer_size: int = 10000
    max_eval_lines: int = 13
    min_eval_lines: int = 1


class Trainer(trainer.Trainer):
    @classmethod
    def args_to_methods(cls):
        mapping = super().args_to_methods()
        mapping["agent_args"] += [our_agent.Agent.__init__]
        mapping["env_args"] += [
            env.Env.__init__,
            trainer.Trainer.make_vec_envs,
        ]
        mapping["run_args"] += [trainer.Trainer.run]
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

    @staticmethod
    def make_env(
        rank: int,
        seed: int,
        env_id=None,
        **kwargs,
    ):
        kwargs.update(rank=rank, random_seed=seed + rank)
        return env.Env(**kwargs)

    # noinspection PyMethodOverriding
    @classmethod
    def make_vec_envs(
        cls,
        curriculum_setting,
        evaluating: bool,
        failure_buffer: Queue,
        max_eval_lines: int,
        min_eval_lines: int,
        max_lines: int,
        min_lines: int,
        world_size: int,
        **kwargs,
    ):
        if evaluating:
            min_lines = min_eval_lines
            max_lines = max_eval_lines
        data_types.WORLD_SIZE = world_size
        mp_kwargs = dict()
        return super().make_vec_envs(
            mp_kwargs=mp_kwargs,
            min_lines=min_lines,
            max_lines=max_lines,
            evaluating=evaluating,
            world_size=world_size,
            failure_buffer=failure_buffer,
            **kwargs,
        )

    @classmethod
    def structure_config(cls, cfg: DictConfig) -> Dict[str, any]:
        if cfg.eval.interval:
            cfg.eval.steps = 5 * cfg.max_eval_lines
        return super().structure_config(cfg)


@hydra.main(config_name="config")
def app(cfg: DictConfig) -> None:
    pprint(dict(**cfg))
    Trainer.main(cfg)


def main(_app):
    cs = ConfigStore.instance()
    cs.store(name="config", node=OurConfig)
    _app()


if __name__ == "__main__":
    main(app)
