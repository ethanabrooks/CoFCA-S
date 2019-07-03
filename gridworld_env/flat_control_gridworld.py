from collections import namedtuple

import numpy as np
from gym import spaces

from gridworld_env.control_flow_gridworld import ControlFlowGridWorld

Obs = namedtuple("Obs", "base subtask subtasks next_subtask lines")


class FlatControlFlowGridWorld(ControlFlowGridWorld):
    def __init__(self, *args, n_subtasks, **kwargs):
        super().__init__(*args, n_subtasks=n_subtasks, **kwargs)
        obs_spaces = self.observation_space.spaces
        subtask_nvec = obs_spaces["subtasks"].nvec[0]
        self.lines = None
        self.observation_space.spaces = Obs(
            base=obs_spaces["base"],
            subtask=obs_spaces["subtask"],
            next_subtask=obs_spaces["next_subtask"],
            subtasks=obs_spaces["subtasks"],
            lines=spaces.MultiDiscrete(
                np.tile(
                    np.pad(
                        subtask_nvec,
                        [0, 1],
                        "constant",
                        constant_values=1 + len(self.object_types),
                    ),
                    (self.n_subtasks + self.n_subtasks // 2, 1),
                )
            ),
        )._asdict()

    def get_observation(self):
        obs = super().get_observation()

        def get_lines():
            for subtask, (pos, neg), condition in zip(
                self.subtasks, self.control, self.conditions
            ):
                yield subtask + (0,)
                if pos != neg:
                    yield (0, 0, 0, condition + 1)

        self.lines = np.vstack(list(get_lines()))
        obs = Obs(
            base=obs["base"],
            subtask=obs["subtask"],
            next_subtask=obs["next_subtask"],
            subtasks=obs["subtasks"],
            lines=self.lines,
        )
        # for (k, s), o in zip(self.observation_space.spaces.items(), obs):
        #     assert s.contains(o)
        return obs._asdict()

    def get_control(self):
        for i in range(self.n_subtasks):
            if i % 2 == 0:
                yield i + 1, i
            else:
                yield i, i


def main(seed, n_subtasks):
    kwargs = gridworld_env.get_args("4x4SubtasksGridWorld-v0")
    del kwargs["class_"]
    del kwargs["max_episode_steps"]
    kwargs.update(n_subtasks=n_subtasks, max_task_count=1)
    env = FlatControlFlowGridWorld(**kwargs, evaluation=False, eval_subtasks=[])
    actions = "wsadeq"
    gridworld_env.keyboard_control.run(env, actions=actions, seed=seed)


if __name__ == "__main__":
    import argparse
    import gridworld_env.keyboard_control

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-subtasks", type=int, default=5)
    main(**vars(parser.parse_args()))
