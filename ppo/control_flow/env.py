import functools
from abc import ABC
from collections import defaultdict, namedtuple
from typing import List, Tuple, Iterator
import numpy as np
from gym.utils import seeding
from gym.vector.utils import spaces
from rl_utils import hierarchical_parse_args, gym
from ppo import keyboard_control
from ppo.utils import RED, RESET, GREEN
from ppo.control_flow.lines import (
    If,
    Else,
    EndIf,
    While,
    EndWhile,
    Subtask,
    Padding,
    Line,
    Loop,
    EndLoop,
)

Obs = namedtuple("Obs", "active lines obs")
Last = namedtuple("Last", "action active reward terminal selected")
State = namedtuple("State", "obs prev ptr term subtask_complete")
Action = namedtuple("Action", "upper lower delta ag dg ptr")


class Env(gym.Env, ABC):
    def __init__(
        self,
        min_lines,
        max_lines,
        flip_prob,
        num_subtasks,
        max_nesting_depth,
        eval_condition_size,
        single_control_flow_type,
        no_op_limit,
        time_to_waste,
        subtasks_only,
        break_on_fail,
        max_loops,
        rank,
        lower_level,
        control_flow_types,
        seed=0,
        eval_lines=None,
        evaluating=False,
    ):
        super().__init__()
        self.lower_level = lower_level
        if Subtask not in control_flow_types:
            control_flow_types.append(Subtask)
        self.control_flow_types = control_flow_types
        self.rank = rank
        self.max_loops = max_loops
        self.break_on_fail = break_on_fail
        self.subtasks_only = subtasks_only
        self.no_op_limit = no_op_limit
        self._eval_condition_size = eval_condition_size
        self._single_control_flow_type = single_control_flow_type
        self.max_nesting_depth = max_nesting_depth
        self.num_subtasks = num_subtasks
        self.time_to_waste = time_to_waste
        self.time_remaining = None
        self.i = 0

        self.loops = None
        self.eval_lines = eval_lines
        self.min_lines = min_lines
        self.max_lines = max_lines
        if evaluating:
            self.n_lines = eval_lines
        else:
            self.n_lines = max_lines
        self.n_lines += 1
        self.random, self.seed = seeding.np_random(seed)
        self.flip_prob = flip_prob
        self.evaluating = evaluating
        self.iterator = None
        self._render = None
        self.action_space = spaces.MultiDiscrete(
            np.array([self.num_subtasks + 1, 2 * self.n_lines, self.n_lines])
        )

        def possible_lines():
            for i in range(num_subtasks):
                yield Subtask(i)
            for i in range(1, max_loops + 1):
                yield Loop(i)
            for line_type in self.line_types:
                if line_type not in (Subtask, Loop):
                    yield line_type(0)

        self.possible_lines = list(possible_lines())
        self.observation_space = spaces.Dict(
            dict(
                obs=spaces.Discrete(2),
                lines=spaces.MultiDiscrete(
                    np.array([len(self.possible_lines)] * self.n_lines)
                ),
                active=spaces.Discrete(self.n_lines + 1),
            )
        )

    @property
    def line_types(self):
        return [If, Else, EndIf, While, EndWhile, EndLoop, Subtask, Padding, Loop]
        # return list(Line.types)

    def reset(self):
        self.i += 1
        self.iterator = self.generator()
        s, r, t, i = next(self.iterator)
        return s

    def step(self, action):
        return self.iterator.send(action)

    def generator(self):
        step = 0
        n = 0
        state_iterator, lines = self.generators()
        state = next(state_iterator)
        actions = []
        program_counter = []

        subtasks_complete = 0
        agent_ptr = 0
        info = {}
        term = False
        action = None
        lower_level_action = None
        while True:
            if state.ptr is not None:
                program_counter.append(state.ptr)
            success = state.ptr is None

            term = term or success or state.term
            if self.lower_level == "train-alone":
                reward = 1 if state.subtask_complete else 0
            else:
                reward = int(success)
            subtasks_complete += state.subtask_complete
            if term:
                if not success and self.break_on_fail:
                    import ipdb

                    ipdb.set_trace()

                info.update(
                    instruction=[self.preprocess_line(l) for l in lines],
                    actions=actions,
                    program_counter=program_counter,
                    success=success,
                )
                if success:
                    info.update(success_line=len(lines))
                else:
                    info.update(success_line=state.prev, failure_line=state.ptr)
                if self.lower_level == "train-alone":
                    lines_attempted = min(len(lines), subtasks_complete + 1)
                    if not (success and subtasks_complete < len(lines)):
                        info.update(
                            cumulative_reward=subtasks_complete,
                            lines_attempted=lines_attempted,
                        )

            info.update(regret=1 if term and not success else 0)

            def render():
                if term:
                    print(GREEN if success else RED)
                indent = 0
                for i, line in enumerate(lines):
                    if i == state.ptr and i == agent_ptr:
                        pre = "+ "
                    elif i == agent_ptr:
                        pre = "- "
                    elif i == state.ptr:
                        pre = "| "
                    else:
                        pre = "  "
                    indent += line.depth_change[0]
                    print(
                        "{:2}{}{}{}".format(i, pre, " " * indent, self.line_str(line))
                    )
                    indent += line.depth_change[1]
                if agent_ptr < len(self.subtasks):
                    print("Selected:", self.subtasks[agent_ptr], agent_ptr)
                print("Action:", action)
                if lower_level_action is not None:
                    print(
                        "Lower Level Action:",
                        self.lower_level_actions[lower_level_action],
                    )
                print("Reward", reward)
                print("Cumulative", subtasks_complete)
                print("Obs:")
                print(RESET)
                self.print_obs(state.obs)

            self._render = render
            obs = self.get_observation(obs=state.obs, active=state.ptr, lines=lines)

            action = (yield obs, reward, term, info)
            if action.size == 1:
                action = Action(upper=0, lower=action, delta=0, ag=0, dg=0, ptr=0)
            actions.extend([int(a) for a in action])
            action = Action(*action)
            action, lower_level_action, agent_ptr, = (
                int(action.upper),
                int(action.lower),
                int(action.ptr),
            )

            info = {}

            if action == self.num_subtasks:
                n += 1
                no_op_limit = 200 if self.evaluating else self.no_op_limit
                if self.no_op_limit is not None and self.no_op_limit < 0:
                    no_op_limit = len(lines)
                if n >= no_op_limit:
                    term = True
            elif state.ptr is not None:
                step += 1
                state = state_iterator.send((action, lower_level_action))

    @property
    def eval_condition_size(self):
        return self._eval_condition_size and self.evaluating

    @property
    def single_control_flow_type(self):
        return self._single_control_flow_type and not self.evaluating

    def choose_line_types(self):
        if self.evaluating:
            assert self.eval_lines is not None
            n_lines = self.eval_lines
        else:
            n_lines = self.random.random_integers(self.min_lines, self.max_lines)
        if self.eval_condition_size:
            line0 = self.random.choice([While, If])
            edge_length = self.random.random_integers(
                self.max_lines, self.eval_lines - 1
            )
            lines = [line0] + [Subtask] * (edge_length - 2)
            lines += [EndWhile if line0 is While else EndIf, Subtask]
        else:
            line_types = self.control_flow_types
            if self.single_control_flow_type:
                line_types = [self.random.choice(self.control_flow_types), Subtask]
            lines = list(
                Line.generate_types(
                    n_lines,
                    remaining_depth=self.max_nesting_depth,
                    random=self.random,
                    legal_lines=line_types,
                )
            )
        return lines

    def assign_line_ids(self, lines):
        for line in lines:
            if line is Subtask:
                yield Subtask(self.random.choice(self.num_subtasks))
            elif line is Loop:
                yield Loop(self.random.randint(1, 1 + self.max_loops))
            else:
                yield line(0)

    def line_generator(self, lines):
        line_transitions = defaultdict(list)
        for _from, _to in self.get_transitions(lines):
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

    @staticmethod
    def get_transitions(lines):
        conditions = []
        for i, line in enumerate(lines):
            yield from line.transitions(i, conditions)

    def generators(self) -> Tuple[Iterator[State], List[Line]]:
        line_types = self.choose_line_types()

        def line_generator():
            for line_type in line_types:
                if line_type is Subtask:
                    line_id = self.random.choice(self.num_subtasks)
                elif line_type is Loop:
                    line_id = self.random.randint(1, 1 + self.max_loops)
                else:
                    line_id = 0
                yield line_type(line_id)

        lines = list(line_generator())  # type List[Line]
        line_transitions = defaultdict(list)
        for _from, _to in self.get_transitions(lines):
            line_transitions[_from].append(_to)

        def line_generator():
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

        def state_generator():
            line_iterator = line_generator()
            condition_bit = 0 if self.eval_condition_size else self.random.choice(2)
            self.time_remaining = self.time_to_waste
            self.loops = None

            def next_subtask(msg=condition_bit):
                l = line_iterator.send(msg)
                while not (l is None or type(lines[l]) is Subtask):
                    line = lines[l]
                    if type(line) is Loop:
                        if self.loops is None:
                            self.loops = line.id
                        else:
                            self.loops -= 1
                        l = line_iterator.send(self.loops > 0)
                        if self.loops == 0:
                            self.loops = None
                    else:
                        l = line_iterator.send(condition_bit)
                self.time_remaining += 1
                return l

            prev, ptr = 0, next_subtask(None)
            term = False
            while True:
                action = yield State(obs=condition_bit, prev=prev, ptr=ptr, term=term)
                if not self.time_remaining or action != lines[ptr].id:
                    term = True
                else:
                    self.time_remaining -= 1
                    condition_bit = abs(
                        condition_bit - int(self.random.rand() < self.flip_prob)
                    )
                    prev, ptr = ptr, next_subtask()

        return state_generator(), lines

    @functools.lru_cache(maxsize=120)
    def preprocess_line(self, line):
        return self.possible_lines.index(line)

    def get_observation(self, obs, active, lines):
        padded = lines + [Padding(0)] * (self.n_lines - len(lines))
        lines = [self.preprocess_line(p) for p in padded]
        obs = Obs(
            obs=obs, lines=lines, active=self.n_lines if active is None else active
        )._asdict()
        # if not self.observation_space.contains(obs):
        #     import ipdb
        #
        #     ipdb.set_trace()
        #     self.observation_space.contains(obs)
        return obs

    @staticmethod
    def print_obs(obs):
        print(obs)

    def seed(self, seed=None):
        assert self.seed == seed

    def render(self, mode="human", pause=True):
        self._render()
        if pause:
            input("pause")

    @staticmethod
    def line_str(line):
        return line


def build_parser(p):
    p.add_argument("--min-lines", type=int, required=True)
    p.add_argument("--max-lines", type=int, required=True)
    p.add_argument("--num-subtasks", type=int, default=12)
    p.add_argument("--max-loops", type=int, default=2)
    p.add_argument("--no-op-limit", type=int)
    p.add_argument("--flip-prob", type=float, default=0.5)
    p.add_argument("--eval-condition-size", action="store_true")
    p.add_argument("--single-control-flow-type", action="store_true")
    p.add_argument("--max-nesting-depth", type=int, default=1)
    p.add_argument("--subtasks-only", action="store_true")
    p.add_argument("--break-on-fail", action="store_true")
    p.add_argument("--time-to-waste", type=int, required=True)
    p.add_argument(
        "--control-flow-types",
        nargs="*",
        type=lambda s: dict(
            Subtask=Subtask, If=If, Else=Else, While=While, Loop=Loop
        ).get(s),
    )


def main(env):
    def action_fn(string):
        ll = dict(w=6, s=12, a=8, d=10, m=0, t=1, o=2, g=3, i=4).get(string, None)
        if ll is None:
            return None
        return Action(upper=0, lower=ll, delta=0, dg=0, ag=0, ptr=0)

    keyboard_control.run(env, action_fn=action_fn)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    main(Env(**hierarchical_parse_args(build_parser(parser))))
