import typing
from dataclasses import astuple, dataclass
from multiprocessing import Queue
from pprint import pprint
from typing import Union, Generator, Tuple, List, Optional

import gym
import hydra
import numpy as np
from colored import fg
from gym import spaces
from gym.utils import seeding
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

import keyboard_control
import osx_queue
from utils import RESET, Discrete


@dataclass
class EnvConfig:
    break_on_fail: bool = False
    max_lines: int = 10
    min_lines: int = 2
    tgt_success_rate: float = 0.75


@dataclass(frozen=True)
class RawAction:
    delta: typing.Any
    pointer: typing.Any

    @staticmethod
    def parse(*xs) -> "RawAction":
        delta, ptr = xs
        return RawAction(delta, ptr)

    def flatten(self) -> Generator[any, None, None]:
        yield from astuple(self)


# noinspection PyAttributeOutsideInit
@dataclass
class Env(gym.Env):
    break_on_fail: bool
    failure_buffer: Queue
    max_lines: int
    min_lines: int
    rank: int
    random_seed: int
    tgt_success_rate: float
    alpha: float = 0.05
    evaluating: bool = None
    i: int = 0
    iterator = None
    render_thunk = None
    success_avg = 0.5
    success_with_failure_buf_avg = 0.5

    def __post_init__(self):
        self.random, _ = seeding.np_random(self.random_seed)
        self.n_lines_space = Discrete(self.min_lines, self.max_lines)
        self.n_lines_space.seed(self.random_seed)
        self.non_failure_random = self.random.get_state()
        self.act_spaces = RawAction(
            delta=2 * self.max_lines,
            pointer=self.max_lines,
        )
        self.action_space = spaces.MultiDiscrete(np.array(astuple(self.act_spaces)))
        self.observation_space = spaces.MultiDiscrete(
            np.array([2] * self.max_lines)  # +1 for padding
        )

    def state_generator(
        self, *instructions
    ) -> Generator[bool, Optional[RawAction], None]:
        def last1() -> int:
            for i, b in reversed(list(enumerate(instructions))):
                if b == 1:
                    return i

        action = None

        def render():
            print(instructions)
            pprint(action)

        self.render_thunk = render
        success = False

        while True:
            action: Optional[RawAction]
            # noinspection PyTypeChecker
            action = yield success, render
            success = action.pointer == last1()
            print("state", success)

    @staticmethod
    def done_generator():
        yield False, lambda: None
        while True:
            # noinspection PyTypeChecker
            yield True, lambda: None

    @staticmethod
    def info_generator():
        success = None
        i = {}
        term = False
        while True:
            # noinspection PyTupleAssignmentBalance
            if term:
                i = dict(success=success)
            print("info", success)
            (success, term) = yield i, lambda: None

    @staticmethod
    def reward_generator():
        success = yield

        while True:
            reward = float(success)
            print("reward", success)
            # noinspection PyTypeChecker
            success = yield reward, lambda: print("Reward:", reward)

    def obs_generator(
        self,
        *lines,
    ):
        padded: List[int] = [
            *(1 + np.array(lines)),
            *[0] * (self.max_lines - len(lines)),
        ]

        def render():
            pprint(lines)

        while True:
            # noinspection PyTypeChecker
            yield padded, lambda: render()  # perform time-step

    def main(self):
        def action_fn():
            while True:
                action = input("go:")
                try:
                    action = int(action)
                    return RawAction.parse(action, action)
                except ValueError:
                    pass

        keyboard_control.run(self, action_fn)

    def render(self, mode="human", pause=True):
        self.render_thunk()
        if pause:
            input("pause")

    def reset(self):
        self.i += 1
        self.iterator = self.srti_generator()
        s, r, t, i = next(self.iterator)
        return s

    def seed(self, seed=None):
        assert self.random_seed == seed

    def srti_generator(
        self,
    ) -> Generator[Tuple[any, float, bool, dict], Optional[RawAction], None]:
        n_lines = self.n_lines_space.sample()
        instructions = self.random.choice(2, size=n_lines)
        instructions[self.random.choice(n_lines)] = 1
        obs_iterator = self.obs_generator(*instructions)
        reward_iterator = self.reward_generator()
        done_iterator = self.done_generator()
        info_iterator = self.info_generator()
        state_iterator = self.state_generator(*instructions)
        next(obs_iterator)
        next(reward_iterator)
        next(done_iterator)
        next(info_iterator)
        state, render_state = next(state_iterator)

        def render():
            if t:
                print("srti", i)
                print(fg("green") if i["success"] else fg("red"))
            render_r()
            render_t()
            render_i()
            render_state()
            render_s()
            print(RESET)

        while True:
            s, render_s = obs_iterator.send(state)
            r, render_r = reward_iterator.send(state)
            t, render_t = done_iterator.send(state)
            i, render_i = info_iterator.send((state, t))

            if self.break_on_fail and t and not i["success"]:
                import ipdb

                ipdb.set_trace()

            self.render_thunk = render

            a: Optional[RawAction]
            # noinspection PyTypeChecker
            a = yield s, r, t, i

            state, render_state = state_iterator.send(a)

    def step(self, action: np.ndarray):
        return self.iterator.send(action)


@hydra.main(config_name="config")
def app(cfg: DictConfig) -> None:
    failure_buffer = Queue()
    try:
        failure_buffer.qsize()
    except NotImplementedError:
        failure_buffer = osx_queue.Queue()
    Env(
        **cfg,
        rank=0,
        failure_buffer=failure_buffer,
    ).main()


if __name__ == "__main__":

    @dataclass
    class Config(EnvConfig):
        random_seed: int = 0

    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
    app()
