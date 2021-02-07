import pickle
from collections import OrderedDict
from dataclasses import asdict, dataclass, astuple
from multiprocessing import Queue
from pathlib import Path
from queue import Full, Empty
from typing import Union, Generator, Tuple, List, Optional

import gym
import hydra
import numpy as np
from colored import fg
from gym.spaces import Discrete, MultiBinary, MultiDiscrete, Box
from gym.utils import seeding
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

import keyboard_control
import osx_queue
from data_types import RawAction, Obs
from minecraft import data_types
from minecraft.data_types import State, Expression, Action, Pad, Line
from utils import RESET


@dataclass
class EnvConfig:
    break_on_fail: bool = False
    bucket_size: int = 5
    check_spaces: bool = False
    max_depth: int = 1
    max_lines: int = 10
    min_lines: int = 1
    num_subtasks: int = 10
    tgt_success_rate: float = 1
    time_per_line: int = 25


@dataclass
class Env(gym.Env):
    break_on_fail: bool
    bucket_size: int
    check_spaces: bool
    failure_buffer: Queue
    max_depth: int
    max_lines: int
    min_lines: int
    num_subtasks: int
    rank: int
    random_seed: int
    tgt_success_rate: float
    time_per_line: int
    alpha: float = 0.05
    evaluating: bool = None
    i: int = 0
    iterator = None
    render_thunk = None
    success_avg = 0.5
    success_with_failure_buf_avg = 0.5

    def __post_init__(self):
        data_types.NUM_SUBTASKS = self.num_subtasks
        self.random, _ = seeding.np_random(self.random_seed)
        self.non_failure_random = self.random.get_state()
        self.render_thunk = None
        self.act_spaces = RawAction(
            delta=2 * self.max_lines,
            gate=2,
            pointer=self.max_lines,
            extrinsic=self.num_subtasks,
        )
        self.obs_spaces = Obs(
            action_mask=MultiBinary(self.num_subtasks),
            destroyed_unit=Discrete(1),
            gate_openers=MultiDiscrete(1 + np.arange(self.num_subtasks).reshape(-1, 1)),
            instructions=MultiDiscrete([Line.space().n] * self.max_lines),
            instruction_mask=MultiBinary(self.max_lines),
            obs=Box(high=1, low=0, shape=(1, 1, 1)),
            partial_action=MultiDiscrete([self.num_subtasks]),
            resources=MultiDiscrete([]),
            pointer=Discrete(self.max_lines),
        )
        self.action_space = MultiDiscrete(np.array(astuple(self.act_spaces)))
        self.observation_space = gym.spaces.Dict(asdict(self.obs_spaces))

    @staticmethod
    def done_generator():
        state: State
        state = yield

        while True:
            # noinspection PyTypeChecker
            state = (
                yield state.success or state.wrong_move or not state.time_remaining,
                lambda: None,
            )

    @staticmethod
    def dump(name: str, x) -> Path:
        path = Path(f"{name}.pkl")
        with path.open("wb") as f:
            pickle.dump(x, f)
        return path.absolute()

    def failure_buffer_wrapper(self, iterator):
        use_failure_buf = False
        size = self.failure_buffer.qsize()
        if self.evaluating or not size:
            use_failure_buf = False
        else:
            success_avg = max(
                self.success_avg, self.success_with_failure_buf_avg + 1e-6
            )
            tgt_success_rate = max(
                self.success_with_failure_buf_avg,
                min(self.tgt_success_rate, success_avg),
            )
            use_failure_prob = 1 - (
                tgt_success_rate - self.success_with_failure_buf_avg
            ) / (success_avg - self.success_with_failure_buf_avg)
            use_failure_buf = self.random.random() < use_failure_prob
        state = None
        if use_failure_buf:

            # randomly rotate queue
            for i in range(self.random.choice(min(100, size))):
                try:
                    state = self.failure_buffer.get_nowait()
                    self.failure_buffer.put_nowait(state)
                except Full:
                    pass  # discard, keep going
                except Empty:
                    break

            try:
                state = self.failure_buffer.get_nowait()
            except (Full, Empty):
                use_failure_buf = state is not None

        if not use_failure_buf:
            state = self.non_failure_random
        self.random.set_state(state)
        initial_random = self.random.get_state()
        action = None

        def render_thunk():
            return

        def render():
            render_thunk()
            if use_failure_buf:
                print(fg("red"), "Used failure buffer", RESET)
            else:
                print(fg("blue"), "Did not use failure buffer", RESET)

        while True:
            s, r, t, i = iterator.send(action)
            render_thunk = self.render_thunk
            self.render_thunk = render
            if t:
                success = i["success"]

                if not self.evaluating:
                    i.update(
                        {
                            f"{k} ({'with' if use_failure_buf else 'without'} failure buffer)": v
                            for k, v in i.items()
                        }
                    )

                def interpolate(old, new):
                    return old + self.alpha * (new - old)

                if use_failure_buf:
                    self.success_with_failure_buf_avg = interpolate(
                        self.success_with_failure_buf_avg, success
                    )
                else:
                    self.success_avg = interpolate(self.success_avg, success)

                put_failure_buf = not self.evaluating and not success
                if put_failure_buf:
                    try:
                        self.failure_buffer.put_nowait(initial_random)
                    except Full:
                        pass

                i.update({"used failure buffer": use_failure_buf})

            if t:
                # noinspection PyAttributeOutsideInit
                self.non_failure_random = self.random.get_state()
            action = yield s, r, t, i

    def info_generator(self, *lines):
        state: State
        done: bool
        state, done = yield
        info = {}
        elapsed_time = -1

        while True:
            if done:
                if self.evaluating:
                    lower = (len(lines) - 1) // self.bucket_size * self.bucket_size + 1
                    upper = (
                        1 + (len(lines) - 1) // self.bucket_size
                    ) * self.bucket_size
                    key = (
                        f"success on instructions length-{lower} through length-{upper}"
                    )
                else:
                    key = f"success on length-{len(lines)} instructions"
                info.update(
                    {
                        f"success": float(state.success),
                        key: float(state.success),
                        "instruction length": len(lines),
                        "time per line": elapsed_time / len(lines),
                    },
                )

            # noinspection PyTupleAssignmentBalance
            state, done = yield info, lambda: None
            info = {}
            elapsed_time += 1

    @staticmethod
    def load(path: str) -> State:
        with Path(path).open("rb") as f:
            return pickle.load(f)

    def main(self):
        def action_fn():
            while True:
                action = input("go:")
                try:
                    action = int(action) + 1
                    return Action.parse(extrinsic=action)
                except ValueError:
                    return Action.parse(extrinsic=0)

        keyboard_control.run(self, action_fn)

    def obs_generator(
        self,
        instructions: Expression,
    ):
        state: State
        state = yield

        instruction_list: List[Line] = [*instructions]
        padded: List[Line] = [
            *instruction_list,
            *[Pad()] * (self.max_lines - len(instruction_list)),
        ]

        def render():
            pass

        while True:
            obs = OrderedDict(
                asdict(
                    Obs(
                        action_mask=np.ones(self.num_subtasks),
                        destroyed_unit=0,
                        gate_openers=np.arange(self.num_subtasks).reshape(-1, 1),
                        instruction_mask=[int(p == Pad()) for p in padded],
                        instructions=[l.to_int() for l in padded],
                        obs=[[[state.condition_bit]]],
                        partial_action=[state.action.to_ints().extrinsic],
                        pointer=state.agent_pointer,
                        resources=[],
                    )
                )
            )
            if self.check_spaces:
                for (k, space), (n, o) in zip(
                    self.observation_space.spaces.items(), obs.items()
                ):
                    if not space.contains(o):
                        import ipdb

                        ipdb.set_trace()
                        space.contains(o)
            # noinspection PyTypeChecker
            state = yield obs, lambda: render()  # perform time-step

    def render(self, mode="human", pause=True):
        self.render_thunk()
        if pause:
            input("pause")

    def reset(self):
        self.i += 1
        self.iterator = self.failure_buffer_wrapper(self.srti_generator())
        s, r, t, i = next(self.iterator)
        return s

    @staticmethod
    def reward_generator():
        state: State
        state = yield

        while True:
            reward = float(
                state.success or not state.time_remaining
            )  # you win if you run down the clock
            # noinspection PyTypeChecker
            state = yield reward, lambda: print("Reward:", reward)

    def seed(self, seed=None):
        assert self.random_seed == seed

    def srti_generator(
        self,
    ) -> Generator[Tuple[any, float, bool, dict], RawAction, None]:
        instructions = Expression.random(self.max_lines, self.random, self.max_depth)
        assert len(instructions) <= self.max_lines
        obs_iterator = self.obs_generator(instructions)
        reward_iterator = self.reward_generator()
        done_iterator = self.done_generator()
        info_iterator = self.info_generator(instructions)
        state_iterator = self.state_generator(instructions)
        next(obs_iterator)
        next(reward_iterator)
        next(done_iterator)
        next(info_iterator)
        state, render_state = next(state_iterator)

        def render():

            if t:
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

            a: RawAction
            # noinspection PyTypeChecker
            a = yield s, r, t, i
            state, render_state = state_iterator.send(a)

    def state_generator(
        self,
        instructions: Expression,
    ) -> Generator[State, Optional[Action], None]:
        action: Optional[Action] = Action.parse(extrinsic=0)
        time_remaining = self.time_per_line * len(instructions)
        wrong_move = False
        success = False
        condition_bit = bool(self.random.choice(2))
        while True:

            def render():
                print("bit:", condition_bit)
                for i, string in enumerate(instructions.strings()):
                    print("-" if i == action.pointer else " ", string, sep="")

            instructions = instructions.set_predicate(condition_bit)

            action: Action
            # noinspection PyTypeChecker
            action = (
                yield State(
                    action=action,
                    agent_pointer=0 if action is None else action.pointer,
                    success=success,
                    wrong_move=wrong_move,
                    condition_bit=condition_bit,
                    time_remaining=time_remaining,
                ),
                render,
            )

            time_remaining -= 1
            if instructions.complete():
                success = True
                continue
            if action.is_op():
                wrong_move = action.extrinsic != instructions.subtask().id
                instructions = instructions.advance()
                condition_bit = bool(self.random.choice(2))

    def step(self, action: Union[np.ndarray, RawAction]):
        if isinstance(action, np.ndarray):
            action = Action.parse(*action)
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
