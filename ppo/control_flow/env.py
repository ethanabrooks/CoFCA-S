import functools
from abc import ABC
from collections import defaultdict, namedtuple, OrderedDict, Counter

import numpy as np

# import skimage.draw
from gym.utils import seeding
from gym.vector.utils import spaces
from rl_utils import hierarchical_parse_args, gym


from ppo import keyboard_control
from ppo.control_flow.lines import (
    If,
    Else,
    EndIf,
    While,
    EndWhile,
    Subtask,
    Padding,
    Line,
)
from ppo.utils import RED, RESET, GREEN

Obs = namedtuple("Obs", "active lines obs")
Last = namedtuple("Last", "action active reward terminal selected")
State = namedtuple("State", "obs condition prev ptr condition_evaluations term")


class Env(gym.Env, ABC):
    pairs = {If: EndIf, Else: EndIf, While: EndWhile}
    line_types = [If, Else, EndIf, While, EndWhile, Subtask, Padding]

    def __init__(
        self,
        min_lines,
        max_lines,
        flip_prob,
        num_subtasks,
        max_nesting_depth,
        eval_condition_size,
        no_op_limit,
        time_to_waste,
        subtasks_only,
        break_on_fail,
        seed=0,
        eval_lines=None,
        evaluating=False,
    ):
        super().__init__()
        self.break_on_fail = break_on_fail
        self.subtasks_only = subtasks_only
        self.no_op_limit = no_op_limit
        self._eval_condition_size = eval_condition_size
        self.max_nesting_depth = max_nesting_depth
        self.num_subtasks = num_subtasks
        self.time_to_waste = time_to_waste
        self.time_remaining = None

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
        self.observation_space = spaces.Dict(
            dict(
                obs=spaces.Discrete(2),
                lines=spaces.MultiDiscrete(
                    np.array([len(self.line_types) + num_subtasks] * self.n_lines)
                ),
                active=spaces.Discrete(self.n_lines + 1),
            )
        )

    def reset(self):
        self.iterator = self.generator()
        s, r, t, i = next(self.iterator)
        return s

    def step(self, action):
        return self.iterator.send(action)

    def generator(self):
        step = 0
        n = 0
        lines = self.build_lines()
        state_iterator = self.state_generator(lines)
        state = next(state_iterator)
        visited_by_env = []
        visited_by_agent = []

        agent_ptr = 0
        info = {}
        term = False
        action = None
        while True:
            success = state.ptr is None
            reward = int(success)
            if success:
                info.update(success_line=len(lines))

            term = success or state.term
            if term and not success:
                if self.break_on_fail:
                    import ipdb

                    ipdb.set_trace()

                info.update(
                    self.analyze_mistakes(
                        agent_ptr=agent_ptr,
                        env_ptr=state.ptr,
                        visited_by_agent=visited_by_agent,
                        visited_by_env=visited_by_env,
                        lines=lines,
                    )
                )

            info.update(regret=1 if term and not success else 0)
            if term:
                info.update(
                    if_evaluations=state.condition_evaluations[If],
                    while_evaluations=state.condition_evaluations[While],
                )

            def line_strings(index, level):
                if index == len(lines):
                    return
                line = lines[index]
                if type(line) in [Else, EndIf, EndWhile]:
                    level -= 1
                if index == state.ptr and index == agent_ptr:
                    pre = "+ "
                elif index == agent_ptr:
                    pre = "- "
                elif index == state.ptr:
                    pre = "| "
                else:
                    pre = "  "
                indent = pre * level
                # if type(line) is Subtask:
                yield f"{indent}{self.line_str(line)}"
                # else:
                #     yield f"{indent}{line.__name__}"
                # if line in [If, While, Else]:
                if type(line) in [If, While, Else]:
                    level += 1
                yield from line_strings(index + 1, level)

            def render():
                if term:
                    print(GREEN if success else RED)
                for i, string in enumerate(line_strings(index=0, level=1)):
                    print(f"{i}{string}")
                print("Selected:", agent_ptr)
                print("Action:", action)
                print("Reward", reward)
                print("Obs:")
                print(RESET)
                self.print_obs(state.obs)

            self._render = render

            action = (
                yield self.get_observation(state.obs, state.ptr, lines),
                reward,
                term,
                info,
            )

            action, agent_ptr = int(action[0]), int(action[-1])
            if action != self.num_subtasks:
                visited_by_agent.append(agent_ptr)
                visited_by_env.append(state.ptr)
            info = self.get_task_info(lines) if step == 0 else {}

            if action == self.num_subtasks:
                n += 1
                no_op_limit = self.no_op_limit
                if self.no_op_limit is not None and self.no_op_limit < 0:
                    no_op_limit = len(lines)
                if not self.evaluating and n == no_op_limit:
                    term = True
            elif state.ptr is not None:
                step += 1
                if action != lines[state.ptr].id:
                    info.update(success_line=state.prev, failure_line=state.ptr)
                state = state_iterator.send(action)

    def analyze_mistakes(
        self, agent_ptr, env_ptr, visited_by_agent, visited_by_env, lines
    ):
        def get_block(l):
            block_type = None
            l1 = None
            for l2, line in enumerate(lines):
                if type(line) in (Else, EndIf, EndWhile) and l1 < l < l2:
                    return block_type, (l1, l2)
                if type(line) in (If, Else, While):
                    block_type = type(line)
                    l1 = l2
            return None, (None, None)

        agent_block, (agent_block_start, agent_block_end) = get_block(agent_ptr)
        env_block, (env_block_start, env_block_end) = get_block(env_ptr)
        info = dict(
            failed_to_enter_if=0,
            failed_to_enter_else=0,
            mistakenly_enterred_if=0,
            mistakenly_enterred_else=0,
            failed_to_reenter_while=0,
            failed_to_enter_while=0,
            mistakenly_reentered_while=0,
            mistakenly_entered_while=0,
            mistakenly_advanced=0,
            failed_to_keep_up=0,
            mistaken_id=0,
        )
        if env_block is If and env_ptr < agent_ptr and env_ptr not in visited_by_agent:
            info.update(failed_to_enter_if=1)
        elif env_block is Else and (
            (env_ptr < agent_ptr and env_ptr not in visited_by_agent)
            or (agent_block_end == env_block_start)
        ):
            info.update(failed_to_enter_else=1)
        elif (
            agent_block is If
            and agent_ptr < env_ptr
            and agent_ptr not in visited_by_env
        ):
            info.update(mistakenly_enterred_if=1)
        elif (
            agent_block is Else
            and agent_ptr not in visited_by_env
            and (agent_ptr < env_ptr or env_block_end == agent_block_start)
        ):
            info.update(mistakenly_enterred_else=1)
        elif env_block is While and env_ptr < agent_ptr:
            if env_ptr in visited_by_agent:
                info.update(failed_to_reenter_while=1)
            else:
                info.update(failed_to_enter_while=1)
        elif agent_block is While and agent_block_end < env_ptr:
            if agent_ptr in visited_by_env:
                info.update(mistakenly_reentered_while=1)
            else:
                info.update(mistakenly_entered_while=1)
        elif env_ptr < agent_ptr:
            info.update(mistakenly_advanced=1)
        elif agent_ptr < env_ptr:
            info.update(failed_to_keep_up=1)
        else:
            info.update(mistaken_id=1)
        return info

    @staticmethod
    def line_str(line: Line):
        return str(line)

    @property
    def eval_condition_size(self):
        return self._eval_condition_size and self.evaluating

    def build_lines(self):
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
            lines = self.get_lines(
                n_lines, active_conditions=[], max_nesting_depth=self.max_nesting_depth
            )
        return list(self.assign_line_ids(lines))

    def assign_line_ids(self, lines):
        for line in lines:
            if line is Subtask:
                yield Subtask(self.random.choice(self.num_subtasks))
            elif line is Padding:
                yield line(0)
            else:
                yield line(self.line_types.index(line))

    def build_task_image(self, lines):
        image = np.zeros(self.image_shape)
        points = np.round(
            (self.image_size - 1) * self.random.random((len(lines) + 1, 2))
        ).astype(int)

        def draw_circle(p, d):
            r, c = skimage.draw.circle(*p, 2)
            r = np.minimum(r, self.image_size - 1)
            c = np.minimum(c, self.image_size - 1)
            image[d, r, c] = 1

        draw_circle(points[0], d=-2)  # mark start
        for point, line in zip(points, lines):
            depth = 2 + min(self.preprocess_line(line), self.num_subtasks)
            draw_circle(point, d=depth)
        draw_circle(points[-1], d=-1)  # mark end

        np.set_printoptions(threshold=10000, linewidth=100000)
        for bit in (0, 1):

            def edges():
                line_iterator = self.line_generator(lines)
                ptr = next(line_iterator)
                while True:
                    prev, ptr = ptr, line_iterator.send(bit)
                    if ptr is None:
                        yield prev, len(lines)
                        return
                    yield prev, ptr
                    if bit and lines[prev] is EndWhile:
                        assert lines[ptr] is While
                        for _ in range(2):
                            ptr = line_iterator.send(False)  # prevent forever loop
                            if ptr is None:
                                return

            for n, (_from, _to) in enumerate(edges()):
                rr, cc, val = skimage.draw.line_aa(*points[_from], *points[_to])
                image[bit, rr, cc] = val * np.linspace(0.1, 1, val.size)

        from PIL import Image
        from matplotlib import cm

        n = image[2:].shape[0]
        grade = np.linspace(0.1, 1, n).reshape((n, 1, 1))
        myarray = (grade * image[2:]).sum(0)
        im = Image.fromarray(np.uint8(cm.gist_gray(myarray) * 255))
        im.save("/tmp/nodes.png")

        myarray = image[0]
        im = Image.fromarray(np.uint8(cm.gist_gray(myarray) * 255))
        im.save("/tmp/false.png")
        myarray = image[1]
        im = Image.fromarray(np.uint8(cm.gist_gray(myarray) * 255))
        im.save("/tmp/truth.png")

        return image

    @functools.lru_cache(maxsize=120)
    def preprocess_line(self, line):
        if line is Padding:
            t = line
        else:
            t = type(line)
        if t is Subtask:
            return line.id
        return self.num_subtasks + self.line_types.index(t)

    def get_lines(
        self, n, active_conditions, last=None, nesting_depth=0, max_nesting_depth=None
    ):
        if n < 0:
            return []
        if n == 0:
            return []
        if n == len(active_conditions):
            lines = [self.pairs[c] for c in reversed(active_conditions)]
            return lines + [Subtask for _ in range(n - len(lines))]
        elif n == 1:
            return [Subtask]
        line_types = [Subtask]
        enough_space = n > len(active_conditions) + 2
        if (
            enough_space
            and (max_nesting_depth is None or nesting_depth < max_nesting_depth)
            and not self.subtasks_only
        ):
            line_types += [If, While]
        if active_conditions and last is Subtask:
            last_condition = active_conditions[-1]
            if last_condition is If:
                line_types += [EndIf]
            if last_condition is If and enough_space:
                line_types += [Else]
            elif last_condition is Else:
                line_types += [EndIf]
            elif last_condition is While:
                line_types += [EndWhile]
        line_type = self.random.choice(line_types)
        if line_type in [If, While]:
            active_conditions = active_conditions + [line_type]
            nesting_depth += 1
        elif line_type is Else:
            active_conditions = active_conditions[:-1] + [line_type]
        elif line_type in [EndIf, EndWhile]:
            active_conditions = active_conditions[:-1]
            nesting_depth -= 1
        get_lines = self.get_lines(
            n - 1,
            active_conditions=active_conditions,
            last=line_type,
            nesting_depth=nesting_depth,
            max_nesting_depth=max_nesting_depth,
        )
        return [line_type] + get_lines

    def line_generator(self, lines):
        line_transitions = defaultdict(list)
        for _from, _to in self.get_transitions(iter(enumerate(lines)), []):
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

    def get_transitions(self, lines_iter, previous):
        while True:  # stops at StopIteration
            try:
                current, line = next(lines_iter)
            except StopIteration:
                return
            if type(line) is EndIf or type(line) is Subtask:
                yield current, current + 1  # False
                yield current, current + 1  # True
            if type(line) is If:
                yield from self.get_transitions(
                    lines_iter, previous + [current]
                )  # from = If
            elif type(line) is Else:
                prev = previous[-1]
                yield prev, current  # False: If -> Else
                yield prev, prev + 1  # True: If -> If + 1
                previous[-1] = current
            elif type(line) is EndIf:
                prev = previous[-1]
                yield prev, current  # False: If/Else -> EndIf
                yield prev, prev + 1  # True: If/Else -> If/Else + 1
                return
            elif type(line) is While:
                yield from self.get_transitions(
                    lines_iter, previous + [current]
                )  # from = While
            elif type(line) is EndWhile:
                prev = previous[-1]
                # While
                yield prev, current + 1  # False: While -> EndWhile + 1
                yield prev, prev + 1  # True: While -> While + 1
                # EndWhile
                yield current, prev  # False: EndWhile -> While
                yield current, prev  # True: EndWhile -> While
                return

    def state_generator(self, lines) -> State:
        line_iterator = self.line_generator(lines)
        condition_bit = 0 if self.eval_condition_size else self.random.randint(0, 2)
        condition_evaluations = defaultdict(list)
        self.time_remaining = self.time_to_waste

        def next_subtask(msg=condition_bit):
            l = line_iterator.send(msg)
            while not (l is None or type(lines[l]) is Subtask):
                line = lines[l]
                if type(line) in (If, While):
                    condition_evaluations[type(line)] += [condition_bit]
                l = line_iterator.send(condition_bit)
            self.time_remaining += 1
            return l

        prev, ptr = 0, next_subtask(None)
        term = False
        while True:
            action = yield State(
                obs=condition_bit,
                condition=condition_bit,
                prev=prev,
                ptr=ptr,
                condition_evaluations=condition_evaluations,
                term=term,
            )
            if not self.time_remaining or action != lines[ptr].id:
                term = True
            else:
                self.time_remaining -= 1
                condition_bit = abs(
                    condition_bit - int(self.random.rand() < self.flip_prob)
                )
                prev, ptr = ptr, next_subtask()

    def get_observation(self, obs, active, lines):
        padded = lines + [Padding(0)] * (self.n_lines - len(lines))
        lines = [self.preprocess_line(p) for p in padded]
        return Obs(
            obs=obs, lines=lines, active=self.n_lines if active is None else active
        )._asdict()

    @staticmethod
    def print_obs(obs):
        print(obs)

    def get_task_info(self, lines):
        num_if = lines.count(If)
        num_else = lines.count(Else)
        num_while = lines.count(While)
        num_subtask = lines.count(lambda l: type(l) is Subtask)
        i = dict(
            if_lines=num_if,
            else_lines=num_else,
            while_lines=num_while,
            nesting_depth=self.get_nesting_depth(lines),
            num_edges=2 * (num_if + num_else + num_while) + num_subtask,
        )
        keys = {
            (If, EndIf): "if clause length",
            (If, Else): "if-else clause length",
            (Else, EndIf): "else clause length",
            (While, EndWhile): "while clause length",
        }
        for k, v in self.average_interval(lines):
            i[keys[k]] = v
        return i

    @staticmethod
    def average_interval(lines):
        intervals = defaultdict(lambda: [None])
        pairs = [(If, EndIf), (While, EndWhile)]
        if Else in lines:
            pairs.extend([(If, Else), (Else, EndIf)])
        for line in lines:
            for start, stop in pairs:
                if line is start:
                    intervals[start, stop][-1] = 0
                if line is stop:
                    intervals[start, stop].append(None)
            for k, (*_, value) in intervals.items():
                if value is not None:
                    intervals[k][-1] += 1
        for keys, values in intervals.items():
            values = [v for v in values if v]
            if values:
                yield keys, sum(values) / len(values)

    @staticmethod
    def get_nesting_depth(lines):
        max_depth = 0
        depth = 0
        for line in lines:
            if type(line) in [If, While]:
                depth += 1
            if type(line) in [EndIf, EndWhile]:
                depth -= 1
            max_depth = max(depth, max_depth)
        return max_depth

    def seed(self, seed=None):
        assert self.seed == seed

    def render(self, mode="human", pause=True):
        self._render()
        if pause:
            input("pause")


def build_parser(p):
    p.add_argument("--min-lines", type=int, required=True)
    p.add_argument("--max-lines", type=int, required=True)
    p.add_argument("--num-subtasks", type=int, default=12)
    p.add_argument("--no-op-limit", type=int)
    p.add_argument("--flip-prob", type=float, default=0.5)
    p.add_argument("--eval-condition-size", action="store_true")
    p.add_argument("--max-nesting-depth", type=int, default=1)
    p.add_argument("--time-to-waste", type=int, required=True)
    p.add_argument("--subtasks-only", action="store_true")
    p.add_argument("--break-on-fail", action="store_true")
    return p


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    args = hierarchical_parse_args(build_parser(parser))

    def action_fn(string):
        try:
            return int(string), 0
        except ValueError:
            return

    keyboard_control.run(Env(**args), action_fn=action_fn)
