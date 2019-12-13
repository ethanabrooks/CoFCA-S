import numpy as np
from gym import spaces
from rl_utils import hierarchical_parse_args

import ppo.control_flow.env
from ppo import keyboard_control
from ppo.control_flow.env import build_parser, State
from ppo.control_flow.lines import While, EndWhile, Subtask


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

    def subtask_str(self, subtask: Subtask):
        i, o = self.unravel_id(subtask.id)
        return f"Subtask {subtask.id}: {self.interactions[i]} {self.targets[o]}"

    def print_obs(self, obs):
        condition = obs[-1].mean()
        obs = obs[:-1].transpose(1, 2, 0).astype(int)
        grid_size = obs.astype(int).sum(-1).max()  # max objects per grid
        chars = [" "] + [o for o, *_ in self.targets + self.non_targets]
        for i, row in enumerate(obs):
            string = ""
            for j, col in enumerate(row):
                number = col * (1 + np.arange(col.size))
                crop = sorted(number, reverse=True)[:grid_size]
                string += "".join(chars[x] for x in crop) + "|"
            print(string)
            print("-" * len(string))
        print("Condition:", condition)

    def state_generator(self, lines) -> State:
        assert self.max_nesting_depth == 1
        objects = self.targets + self.non_targets
        ice = objects.index("ice")
        agent_pos = self.random.randint(0, self.world_size, size=2)
        agent_id = objects.index("agent")

        def build_world(condition_bit):
            world = np.zeros(self.world_shape)
            for o, p in object_pos + [(agent_id, agent_pos)]:
                world[tuple((o, *p))] = 1
            world[-1] = condition_bit
            return world

        state_iterator = super().state_generator(lines)
        ids = [self.unravel_id(line.id) for line in lines if type(line) is Subtask]
        positions = self.random.randint(0, self.world_size, size=(len(ids), 2))
        object_pos = [(o, tuple(pos)) for (i, o), pos in zip(ids, positions)]
        state = next(state_iterator)
        while True:
            subtask_id = yield state._replace(obs=build_world(state.condition))
            ac, ob = self.unravel_id(subtask_id)
            pair = ob, tuple(agent_pos)
            correct_id = subtask_id == lines[state.curr].id
            if pair in object_pos:  # standing on the desired object
                if self.interactions[ac] == "pickup":
                    object_pos.remove(pair)
                elif self.interactions[ac] == "transform":
                    object_pos.remove(pair)
                    object_pos.append((ice, tuple(agent_pos)))
                if correct_id:
                    state = next(state_iterator)
            else:
                candidates = [np.array(p) for o, p in object_pos if o == ob]
                if candidates:
                    nearest = min(candidates, key=lambda k: np.sum(agent_pos - k))
                    agent_pos += np.clip(nearest - agent_pos, -1, 1)
                elif correct_id:
                    state = next(state_iterator)

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
