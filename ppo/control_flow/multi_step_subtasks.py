import numpy as np
from gym import spaces
from rl_utils import hierarchical_parse_args

import ppo.control_flow.env
from ppo import keyboard_control
from ppo.control_flow.env import build_parser, State
from ppo.control_flow.lines import Subtask, While, EndWhile


class Env(ppo.control_flow.env.Env):
    targets = ["pig", "sheep", "cat", "greenbot"]
    non_targets = ["ice", "agent"]
    interactions = ["visit", "pickup", "transform"]

    def __init__(self, world_size, num_subtasks, **kwargs):
        assert num_subtasks == len(self.targets) * len(self.interactions)
        super().__init__(num_subtasks=num_subtasks, **kwargs)
        self.world_size = world_size
        self.world_shape = (
            len(self.targets + self.non_targets) + 1,  # last channel for condition
            self.world_size,
            self.world_size,
        )
        self.observation_space.spaces.update(
            obs=spaces.Box(low=0, high=1, shape=self.world_shape)
        )

    def print_obs(self, obs):
        condition = obs[-1].mean()
        obs = obs[:-1].transpose(1, 2, 0).astype(int)
        grid_size = obs.astype(int).sum(-1).max()  # max objects per grid
        chars = [" "] + [o for o, *_ in self.targets + self.non_targets]
        for i, row in enumerate(obs):
            string = ""
            for j, col in enumerate(row):
                cropped = sorted(col * (1 + np.arange(col.size)), reverse=True)[
                    :grid_size
                ]
                string += "".join(chars[x] for x in cropped) + "|"
            print(string)
            print("-" * len(string))
        print("Condition:", condition)

    def state_generator(self, lines) -> State:
        assert self.max_nesting_depth == 1
        objects = self.targets + self.non_targets
        ice = objects.index("ice")
        agent_pos = self.random.randint(0, self.world_size, size=2)
        agent_id = objects.index("agent")

        def assign_positions(bit):
            line_iterator = self.line_generator(lines)
            prev, curr = 0, next(line_iterator)
            while True:
                if type(lines[curr]) is Subtask:
                    _, o = self.unravel_id(lines[curr].id)
                    p = self.random.randint(0, self.world_size, size=2)
                    yield o, tuple(p)
                prev, curr = curr, line_iterator.send(bit)
                if curr is None:
                    return
                if bit and lines[prev] is EndWhile:
                    assert lines[curr] is While
                    for _ in range(2):
                        # prevent forever loop
                        prev, curr = curr, line_iterator.send(False)
                        if curr is None:
                            return

        def build_world():
            world = np.zeros(self.world_shape)
            for o, p in positions + [(agent_id, agent_pos)]:
                world[tuple((o, *p))] = 1
            world[-1] = condition_bit
            return world

        condition_iterator = super().state_generator(lines)
        positions = list(assign_positions(True)) + list(assign_positions(False))
        condition_bit = next(condition_iterator).condition
        done = False
        while True:
            subtask_id = yield State(
                obs=build_world(), condition=condition_bit, done=done
            )
            done = False
            act, tgt = self.unravel_id(subtask_id)
            pair = tgt, tuple(agent_pos)
            if pair in positions:  # standing on the desired object
                if self.interactions[act] == "pickup":
                    positions.remove(pair)
                elif self.interactions[act] == "transform":
                    positions.remove(pair)
                    positions.append((ice, tuple(agent_pos)))
                condition_bit, _, _ = next(condition_iterator)
                done = True
            else:
                candidates = [np.array(p) for o, p in positions if o == tgt]
                if candidates:
                    nearest = min(candidates, key=lambda k: np.sum(agent_pos - k))
                    agent_pos += np.clip(nearest - agent_pos, -1, 1)

    def unravel_id(self, subtask_id):
        i = subtask_id // len(self.targets)
        o = subtask_id % len(self.targets)
        return i, o


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser = build_parser(parser)
    parser.add_argument("--world-size", default=4, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = hierarchical_parse_args(parser)

    def action_fn(string):
        try:
            return int(string), 0
        except ValueError:
            return

    keyboard_control.run(Env(**args, baseline=False), action_fn=action_fn)
