from collections import Counter, namedtuple

from gym import spaces
import numpy as np

from gridworld_env import SubtasksGridWorld

Obs = namedtuple("Obs", "base subtask subtasks conditions control next_subtask pred")


class ControlFlowGridWorld(SubtasksGridWorld):
    def __init__(self, *args, n_subtasks, single_condition=False, **kwargs):
        super().__init__(*args, n_subtasks=n_subtasks, **kwargs)
        self.single_condition = single_condition
        if single_condition:
            assert n_subtasks == 2

        self.conditions = None
        self.control = None
        self.required_objects = None
        obs_spaces = self.observation_space.spaces
        obs_spaces.update(
            conditions=spaces.MultiDiscrete(
                np.array([len(self.object_types)]).repeat(self.n_subtasks)
            ),
            pred=spaces.Discrete(2),
            control=spaces.MultiDiscrete(
                np.tile(
                    np.array([[self.n_subtasks]]),
                    [self.n_subtasks, 2],  # binary conditions
                )
            ),
        )
        self.observation_space = spaces.Dict(Obs(**obs_spaces)._asdict())
        self.pred = None

    def render_current_subtask(self):
        if self.subtask_idx == 0:
            print("none")
        else:
            super().render_current_subtask()

    def render_task(self):
        def helper(i, indent):
            neg, pos = self.control[i]
            condition = self.conditions[i]

            def develop_branch(j, add_indent):
                new_indent = indent + add_indent
                if j == 0:
                    subtask = f""
                else:
                    try:
                        subtask = f"{j}:{self.subtasks[j]}"
                    except IndexError:
                        return f"{new_indent}terminate"
                return f"{new_indent}{subtask}\n{helper(j, new_indent)}"

            if pos == neg:
                return f"{develop_branch(pos, '')}"
            else:
                return f"""\
{indent}if {self.object_types[condition]}:
{develop_branch(pos, '    ')}
{indent}else:
{develop_branch(neg, '    ')}
"""

        print(helper(i=0, indent=""))

    def get_observation(self):
        obs = super().get_observation()
        obs.update(
            control=self.control,
            conditions=self.conditions,
            pred=self.evaluate_condition(),
        )
        return Obs(**obs)._asdict()

    def subtasks_generator(self):
        choices = self.np_random.choice(
            len(self.possible_subtasks), size=self.n_subtasks
        )
        subtasks = [self.Subtask(*self.possible_subtasks[i]) for i in choices]
        i = 0
        encountered = Counter(passing=[], failing=[], subtasks=[])
        while i < self.n_subtasks:
            condition = self.conditions[i]
            passing = condition in self.required_objects
            branching = self.control[i, 0] != self.control[i, 1]
            encountered.update(passing=[condition if branching and passing else None])
            encountered.update(
                failing=[condition if branching and not passing else None]
            )
            i = self.control[i, int(passing)]
            encountered.update(subtasks=[i])

        self.required_objects = list(
            set(o for o in encountered["passing"] if o is not None)
        )
        available = [x for x in self.required_objects]

        for i, subtask in enumerate(subtasks):
            if i in encountered["subtasks"]:
                obj = subtask.object
                to_be_removed = subtask.interaction in {1, 2}
                if obj not in available:
                    if (not to_be_removed and obj in encountered["failing"]) or (
                        to_be_removed and obj in encountered["failing"][: i + 1]
                    ):

                        # choose a different object
                        obj = self.np_random.choice(self.required_objects)
                        subtasks[i] = subtask._replace(object=obj)

                    # add object to map
                    self.required_objects += [obj]
                    available += [obj]

                if to_be_removed:
                    available.remove(obj)

        yield from subtasks

    def reset(self):
        def get_control():
            for i in range(self.n_subtasks):
                j = 2 * i
                if self.force_branching or self.np_random.rand() < 0.7:
                    yield j, j + 1
                else:
                    yield self.np_random.randint(
                        i,
                        self.n_subtasks + (i > 0),  # prevent termination on first turn
                        size=2)

        self.control = 1 + np.minimum(np.array(list(get_control())), self.n_subtasks)
        passing = self.np_random.choice(2, size=self.n_subtasks)
        self.conditions = self.np_random.choice(
            len(self.object_types), size=self.n_subtasks
        )
        self.required_objects = self.conditions[passing]
        self.subtask_idx = 0
        self.pred = self.evaluate_condition()
        self.subtask_idx = self.get_next_subtask()
        self.count = self.subtask.count
        return o._replace(conditions=self.conditions, control=self.control)

    def get_next_subtask(self):
        resolution = self.evaluate_condition()
        return self.control[self.subtask_idx, int(resolution)]

    def evaluate_condition(self):
        object_type = self.conditions[self.subtask_idx]
        return object_type in self.objects.values()

    def get_required_objects(self, _):
        required_objects = list(super().get_required_objects(self.subtasks))
        yield from required_objects
        # for condition in self.conditions:
        # if condition not in required_objects:
        # if self.np_random.rand() < .5:
        # yield condition


def main(seed, n_subtasks):
    kwargs = gridworld_env.get_args("4x4SubtasksGridWorld-v0")
    del kwargs["class_"]
    del kwargs["max_episode_steps"]
    kwargs.update(n_subtasks=n_subtasks, max_task_count=1)
    env = ControlFlowGridWorld(**kwargs, evaluation=False, eval_subtasks=[])
    actions = "wsadeq"
    gridworld_env.keyboard_control.run(env, actions=actions, seed=seed)


if __name__ == "__main__":
    import argparse
    import gridworld_env.keyboard_control

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n-subtasks", type=int)
    main(**vars(parser.parse_args()))
