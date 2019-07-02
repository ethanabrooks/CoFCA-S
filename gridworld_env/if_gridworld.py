import numpy as np
from gym import spaces

from gridworld_env.control_flow_gridworld import ControlFlowGridWorld


class IfGridWorld(ControlFlowGridWorld):
    def __init__(self, *args, n_subtasks, **kwargs):
        super().__init__(*args, n_subtasks=n_subtasks, **kwargs)
        subtasks_nvec = self.observation_space.spaces["subtasks"].nvec
        self.observation_space.spaces.update(
            subtasks=spaces.MultiDiscrete(
                np.pad(
                    subtasks_nvec,
                    [0, 1],
                    "constant",
                    constant_values=len(self.object_types),
                )
            )
        )

    def get_observation(self):
        obs = super().get_observation()

        def get_subtasks():
            for subtask, (pos, neg), condition in zip(
                self.subtasks, self.control, self.conditions
            ):
                yield np.append(subtask, 0)
                if pos != neg:
                    yield np.append(np.zeros(3), condition)

        obs.update(subtasks=np.vstack(get_subtasks()))
        print("if obs")
        print(obs["subtasks"])
        return obs

    def get_control(self):
        for i in range(self.n_subtasks):
            if i % 2 == 0:
                yield i + 1, i + 2
            else:
                yield i + 1, i + i


def main(seed, n_subtasks):
    kwargs = gridworld_env.get_args("4x4SubtasksGridWorld-v0")
    del kwargs["class_"]
    del kwargs["max_episode_steps"]
    kwargs.update(n_subtasks=n_subtasks, max_task_count=1)
    env = IfGridWorld(**kwargs, evaluation=False, eval_subtasks=[])
    actions = "wsadeq"
    gridworld_env.keyboard_control.run(env, actions=actions, seed=seed)


if __name__ == "__main__":
    import argparse
    import gridworld_env.keyboard_control

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-subtasks", type=int, default=5)
    main(**vars(parser.parse_args()))
