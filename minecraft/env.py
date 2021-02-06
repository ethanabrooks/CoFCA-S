import pickle
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from itertools import zip_longest
from multiprocessing import Queue
from pathlib import Path
from pprint import pprint
from queue import Full, Empty
from typing import Union, Generator, Tuple, List, Optional

import gym
import hydra
import numpy as np
from colored import fg
from gym.utils import seeding
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

import keyboard_control
from minecraft import data_types
from minecraft.data_types import State, Expression, Line, Action
from data_types import RawAction
from utils import RESET


@dataclass
class Env(gym.Env):
    break_on_fail: bool
    bucket_size: int
    check_spaces: bool
    failure_buffer: Queue
    max_lines: int
    min_lines: int
    rank: int
    random_seed: int
    tgt_success_rate: float
    time_per_line: int
    world_size: int
    alpha: float = 0.05
    evaluating: bool = None
    i: int = 0
    iterator = None
    render_thunk = None
    success_avg = 0.5
    success_with_failure_buf_avg = 0.5

    def __post_init__(self):
        data_types.WORLD_SIZE = self.world_size
        self.random, _ = seeding.np_random(self.random_seed)

    @staticmethod
    def done_generator():
        state: State
        state = yield

        while True:
            # noinspection PyTypeChecker
            state = (
                yield state.success or not state.time_remaining,
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
        keyboard_control.run(self, lambda: None)

    def obs_generator(
        self,
        *lines: Expression,
    ):
        state: State
        state = yield

        padded: List[Optional[Expression]] = [
            *lines,
            *[None] * (self.max_lines - len(lines)),
        ]

        def render():
            for i, line in enumerate(lines):
                print(
                    "{:2}{}{} {}".format(
                        i,
                        "-" if i == state.agent_pointer else " ",
                        "*"
                        if line in [*required_buildings, *state.required_units]
                        else " ",
                        repr(line),
                    )
                )
            print("Obs:")
            for string in self.room_strings(world):
                print(string, end="")

        def coords():
            yield from state.positions.items()
            for p, b in state.building_positions.items():
                yield b, p

        while True:
            world = np.zeros(self.obs_spaces.obs.shape)
            for o, p in coords():
                world[(WorldObjects.index(o), *p)] = 1
            for p in state.destroyed_buildings.keys():
                world[(-1, *p)] = 1
            assert isinstance(state.action, ActionStage)

            gate_openers: np.ndarray = self.obs_spaces.gate_openers.nvec.copy().reshape(
                -1, CompoundAction.input_space().nvec.size
            )
            gate_openers -= 1
            unpadded_gate_openers = state.action.gate_openers()
            gate_openers[: len(unpadded_gate_openers)] = unpadded_gate_openers

            destroyed_unit = (
                0 if state.destroyed_unit is None else 1 + state.destroyed_unit.to_int()
            )
            obs = OrderedDict(
                asdict(
                    Obs(
                        action_mask=state.action.mask(unit_dependencies).ravel(),
                        destroyed_unit=destroyed_unit,
                        gate_openers=gate_openers.ravel(),
                        instruction_mask=(np.array([int(p is None) for p in padded])),
                        instructions=(np.array([*map(self.preprocess_line, padded)])),
                        obs=world,
                        partial_action=(np.array([*state.action.to_ints()])),
                        ptr=np.clip(state.agent_pointer, 0, self.max_lines - 1),
                        resources=(np.array([state.resources[r] for r in Resource])),
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

    def place_objects(
        self, n_lines: int
    ) -> Generator[Tuple[WorldObject, np.ndarray], None, None]:
        nexus = self.random.choice(self.world_size, size=2)
        yield Nexus(), nexus
        for w in Worker:
            yield w, nexus
        resource_offsets = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        resource_locations = [
            *filter(
                self.world_space.contains,
                nexus + resource_offsets,
            )
        ]
        minerals, gas = self.random.choice(
            len(resource_locations), size=2, replace=False
        )
        minerals = resource_locations[minerals]
        gas = resource_locations[gas]
        yield Resource.MINERALS, minerals
        yield Resource.GAS, gas

        occupied = [nexus, minerals, gas]
        occupied_indices = np.sort(
            np.ravel_multi_index(np.stack(occupied, axis=-1), self.world_shape)
        )

        max_initial_buildings = max(0, (self.world_size ** 2 - len(occupied) - n_lines))
        if max_initial_buildings > 0:
            num_initial_buildings = self.random.randint(max_initial_buildings + 1)
            initial_index = self.random.choice(
                self.world_size ** 2 - len(occupied),
                size=num_initial_buildings,
                replace=False,
            )
            for i in occupied_indices:
                initial_index[initial_index >= i] += 1
            initial_pos = np.stack(
                np.unravel_index(initial_index, self.world_shape), axis=-1
            )
            initial_buildings = self.random.choice(
                Buildings,
                size=num_initial_buildings,
            )
            for b, p in zip(initial_buildings, initial_pos):
                # assert not any(np.array_equal(p, p_) for p_ in occupied)
                # occupied += [p]
                yield b, gas if isinstance(b, Assimilator) else p

    @staticmethod
    def preprocess_line(line: Optional[Expression]) -> int:
        if line is None:
            return 0
        if isinstance(line, Building):
            return 1 + line.to_int()
        if isinstance(line, Unit):
            return 1 + Building.space().n + line.to_int()
        raise RuntimeError

    def render(self, mode="human", pause=True):
        self.render_thunk()
        if pause:
            input("pause")

    def reset(self):
        self.i += 1
        self.iterator = self.failure_buffer_wrapper(self.srti_generator())
        s, r, t, i = next(self.iterator)
        return s

    def room_strings(self, room):
        max_symbol_size = max(
            [
                len(multi_worker_symbol(len(Worker))),
                *[len(strip_color(str(x.symbol))) for x in WorldObjects],
            ]
        )
        max_symbols_per_grid = 3
        for i, row in enumerate(room.transpose((1, 2, 0)).astype(int)):
            for j, channel in enumerate(row):
                (nonzero,) = channel[: len(WorldObjects)].nonzero()
                objects = [WorldObjects[k] for k in nonzero]
                worker_symbol = None
                if len(objects) > max_symbols_per_grid:
                    worker_symbol = f"w{sum([isinstance(o, Worker) for o in objects])}"
                    objects = [o for o in objects if not isinstance(o, Worker)]
                symbols = [o.symbol for o in objects]
                if worker_symbol is not None:
                    symbols += [worker_symbol]

                for _, symbol in zip_longest(range(max_symbols_per_grid), symbols):
                    if symbol is None:
                        symbol = " " * max_symbol_size
                    else:
                        symbol += " " * (max_symbol_size - len(strip_color(symbol)))
                    yield from symbol
                yield RESET
                yield "|"
            grid_size = max_symbols_per_grid * max_symbol_size
            yield f"\n" + ("-" * (grid_size) + "+") * self.world_size + "\n"

    @staticmethod
    def reward_generator():
        state: State
        state = yield

        while True:
            reward = float(state.success)
            # noinspection PyTypeChecker
            state = yield reward, lambda: print("Reward:", reward)

    def seed(self, seed=None):
        assert self.random_seed == seed

    def srti_generator(
        self,
    ) -> Generator[Tuple[any, float, bool, dict], Optional[RawAction], None]:
        (
            instructions,
            building_dependencies,
            unit_dependencies,
        ) = self.build_instructions_and_dependencies()
        assert len(instructions) >= 2
        obs_iterator = self.obs_generator(
            *instructions,
            building_dependencies=building_dependencies,
            unit_dependencies=unit_dependencies,
        )
        reward_iterator = self.reward_generator()
        done_iterator = self.done_generator()
        info_iterator = self.info_generator(*instructions)
        state_iterator = self.state_generator(
            *instructions,
            building_dependencies=building_dependencies,
            unit_dependencies=unit_dependencies,
        )
        next(obs_iterator)
        next(reward_iterator)
        next(done_iterator)
        next(info_iterator)
        state, render_state = next(state_iterator)

        def render():
            # for tree in self.build_trees(building_dependencies):
            #     tree.show()

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

            a: Optional[RawAction]
            # noinspection PyTypeChecker
            a = yield s, r, t, i

            state, render_state = state_iterator.send(a)

    @staticmethod
    def line_generator(lines):
        line_transitions = defaultdict(list)

        def get_transitions():
            conditions = []
            for i, line in enumerate(lines):
                yield from line.transitions(i, conditions)

        for _from, _to in get_transitions():
            line_transitions[_from].append(_to)
        i = 0
        if_evaluations = []
        while True:
            condition_bit = yield None if i >= len(lines) else i
            if type(lines[i]) is Else:
                evaluation = not if_evaluations.pop()
            else:
                evaluation = bool(condition_bit)
            if type(lines[i]) is If:
                if_evaluations.append(evaluation)
            i = line_transitions[i][evaluation]

    def generator(self):
        while True:
            n_lines = self.random.random_integers(self.min_lines, self.max_lines)
            if self.long_jump:
                assert self.evaluating
                len_jump = self.random.randint(
                    self.min_eval_lines - 3, self.max_eval_lines - 3
                )
                use_if = self.random.random() < 0.5
                line_types = [
                    If if use_if else While,
                    *(Subtask for _ in range(len_jump)),
                    EndIf if use_if else EndWhile,
                    Subtask,
                ]
            elif self.single_control_flow_type and self.evaluating:
                assert n_lines >= 6
                while True:
                    line_types = list(
                        Expression.generate_types(
                            n_lines,
                            remaining_depth=self.max_nesting_depth,
                            random=self.random,
                            legal_lines=self.control_flow_types,
                        )
                    )
                    criteria = [
                        Else in line_types,  # Else
                        While in line_types,  # While
                        line_types.count(If) > line_types.count(Else),  # If
                    ]
                    if sum(criteria) >= 2:
                        break
            else:
                legal_lines = (
                    [
                        self.random.choice(
                            list(set(self.control_flow_types) - {Subtask})
                        ),
                        Subtask,
                    ]
                    if (self.single_control_flow_type and not self.evaluating)
                    else self.control_flow_types
                )

                line_types = list(
                    Expression.generate_types(
                        n_lines,
                        remaining_depth=self.max_nesting_depth,
                        random=self.random,
                        legal_lines=legal_lines,
                    )
                )
            lines = list(self.assign_line_ids(line_types))
            assert self.max_nesting_depth == 1
            result = self.populate_world(lines)
            if result is not None:
                _agent_pos, objects = result
                break
        state_iterator = self.state_generator(objects, _agent_pos, lines)
        state = next(state_iterator)

        subtasks_complete = 0
        agent_ptr = 0
        info = {}
        term = False

        lower_level_action = None
        actions = VariableActions()
        agent_ptr = 0
        while True:
            success = state.ptr is None
            self.success_count += success

            term = term or success or state.term
            reward = int(success)
            subtasks_complete += state.subtask_complete
            if term:
                if not success and self.break_on_fail:
                    import ipdb

                    ipdb.set_trace()

                info.update(
                    instruction_len=len(lines),
                )
                if not use_failure_buf:
                    info.update(success_without_failure_buf=success)
                if success:
                    info.update(success_line=len(lines), progress=1)
                else:
                    info.update(
                        success_line=state.prev, progress=state.prev / len(lines)
                    )
                subtasks_attempted = subtasks_complete + (not success)
                info.update(
                    subtasks_complete=subtasks_complete,
                    subtasks_attempted=subtasks_attempted,
                )

            info.update(
                subtask_complete=state.subtask_complete,
            )

            def render():
                self.render_instruction(
                    term=term,
                    success=success,
                    lines=lines,
                    state=state,
                    agent_ptr=agent_ptr,
                )
                self.render_world(
                    state=state,
                    action=actions,
                    reward=reward,
                )

            self._render = render
            obs = state.obs
            pads = [Padding(0)] * (self.n_lines - len(lines))
            padded = lines + pads
            preprocessed_lines = [self.preprocess_line(p) for p in padded]
            mask = [int(not isinstance(l, Padding)) for l in padded]
            truthy = [
                self.evaluate_line(l, None, state.counts)
                if agent_ptr < len(lines)
                else 2
                for l in lines
            ]
            truthy = [2 if t is None else int(t) for t in truthy]
            truthy += [3] * (self.n_lines - len(truthy))

            assert issubclass(actions.active, PartialAction)
            obs = Obs(
                action_mask=[*actions.mask(self.action_nvec.a)],
                active=self.n_lines if state.ptr is None else state.ptr,
                can_open_gate=[*actions.active.is_complete(self.action_nvec.a)],
                lines=preprocessed_lines,
                mask=mask,
                obs=[[obs]],
                inventory=self.inventory_representation(state),
                subtask_complete=state.subtask_complete,
                truthy=truthy,
                partial_action=[*actions.partial_actions()],
            )
            obs = OrderedDict(obs._asdict())
            # for k, v in self.observation_space.spaces.items():
            #     if not v.contains(obs[k]):
            #         import ipdb
            #
            #         ipdb.set_trace()
            #         v.contains(obs[k])

            line_specific_info = {
                f"{k}_{10 * (len(lines) // 10)}": v for k, v in info.items()
            }
            raw_action = (yield obs, reward, term, dict(**info, **line_specific_info))
            raw_action = RawAction(*raw_action)
            actions = actions.update(raw_action.a)
            agent_ptr = raw_action.pointer

            info = dict(
                use_failure_buf=use_failure_buf,
                len_failure_buffer=len(self.failure_buffer),
            )

            if actions.no_op():
                n += 1
                no_op_limit = 200 if self.evaluating else self.no_op_limit
                if self.no_op_limit is not None and self.no_op_limit < 0:
                    no_op_limit = len(lines)
                if n >= no_op_limit:
                    term = True
            elif state.ptr is not None:
                step += 1
                # noinspection PyUnresolvedReferences
                state = state_iterator.send(
                    (actions.verb(), actions.noun(), lower_level_action)
                )

    def state_generator(
        self,
        *instructions: Line,
    ) -> Generator[State, Optional[Action], None]:
        pointer = 0
        time_remaining = self.time_per_line * len(instructions)
        action = Action(delta=0, gate=1, pointer=0, extrinsic=None)
        while True:
            success = pointer >= len(instructions)
            failure = action.is_op and action.extrinsic != instructions[pointer].id

            action: Optional[Action]
            # noinspection PyTypeChecker
            action = yield State(
                action=None,
                env_pointer=pointer,
                agent_pointer=action.pointer,
                success=success,
                failure=failure,
                time_remaining=time_remaining,
            )
            if action is None:
                raise NotImplementedError

    def step(self, action: Union[np.ndarray, ActionStage]):
        if isinstance(action, np.ndarray):
            action = RawAction.parse(*action)
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
