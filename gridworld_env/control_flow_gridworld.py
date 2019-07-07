from collections import Counter, namedtuple

from gym import spaces
import numpy as np

from gridworld_env import SubtasksGridWorld

Obs = namedtuple("Obs", "base subtask subtasks conditions control next_subtask pred")


class ControlFlowGridWorld(SubtasksGridWorld):
    def __init__(self, *args, n_subtasks, force_branching=False, **kwargs):
        super().__init__(*args, n_subtasks=n_subtasks, **kwargs)
        self.pred = None
        self.force_branching = force_branching
        if force_branching:
            assert n_subtasks % 2 == 0

        self.conditions = None
        self.control = None
        self.required_objects = None
        self.observation_space = spaces.Dict(
            Obs(
                **self.observation_space.spaces,
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
            )._asdict()
        )

    def render_task(self):
        def helper(i, indent):
            try:
                subtask = f"{i}:{self.subtasks[i]}"
            except IndexError:
                return f"{indent}terminate"
            neg, pos = self.control[i]
            condition = self.conditions[i]

            # def develop_branch(j, add_indent):
            # new_indent = indent + add_indent
            # try:
            # subtask = f"{j}:{self.subtasks[j]}"
            # except IndexError:
            # return f"{new_indent}terminate"
            # return f"{new_indent}{subtask}\n{helper(j, new_indent)}"

            if pos == neg:
                if_condition = helper(pos, indent)
            else:
                if_condition = f"""\
{indent}if {self.object_types[condition]}:
{helper(pos, indent + '    ')}
{indent}else:
{helper(neg, indent + '    ')}
"""
            return f"{indent}{subtask}\n{if_condition}"

        print(helper(i=0, indent=""))

    def get_observation(self):
        obs = super().get_observation()
        obs.update(control=self.control, conditions=self.conditions, pred=self.pred)
        return Obs(**obs)._asdict()

    # noinspection PyTypeChecker
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
            encountered.update(subtasks=[i])
            i = self.control[i, int(passing)]

        object_types = Counter(range(len(self.object_types)))
        self.required_objects = list(set(encountered["passing"]) - {None})
        available = Counter(self.required_objects)
        for l in encountered.values():
            l.reverse()

        for t, subtask_idx in enumerate(encountered["subtasks"]):
            subtask = subtasks[subtask_idx]
            obj = subtask.object
            to_be_removed = self.interactions[subtask.interaction] in {
                "pick-up",
                "transform",
            }

            def available_now():
                if to_be_removed:
                    required_for_future = Counter(set(encountered["passing"][t:]))
                    return available - required_for_future
                else:
                    return available

            while not available_now()[obj]:
                if to_be_removed:
                    prohibited = Counter(encountered["failing"][:t])
                else:
                    prohibited = Counter(encountered["failing"])
                if obj in prohibited:
                    obj = self.np_random.choice(list(object_types - prohibited))
                    subtasks[subtask_idx] = subtask._replace(object=obj)
                else:
                    available[obj] += 1
                    self.required_objects += [obj]

            if to_be_removed:
                available[obj] -= 1

        yield from subtasks

    def get_control(self):
        for i in range(self.n_subtasks):
            j = 2 * i
            if self.force_branching or self.np_random.rand() < 0.7:
                yield j, j + 1
            else:
                yield j, j

    def reset(self):
        self.control = np.minimum(
            1 + np.array(list(self.get_control())), self.n_subtasks
        )
        n_object_types = self.np_random.randint(1, len(self.object_types))
        object_types = np.arange(len(self.object_types))
        existing = self.np_random.choice(object_types, size=n_object_types)
        non_existing = np.array(list(set(object_types) - set(existing)))
        n_passing = self.np_random.choice(self.n_subtasks)
        n_failing = self.n_subtasks - n_passing
        passing = self.np_random.choice(existing, size=n_passing)
        failing = self.np_random.choice(non_existing, size=n_failing)
        self.conditions = np.concatenate([passing, failing])
        self.np_random.shuffle(self.conditions)
        self.required_objects = passing
        self.pred = False
        return super().reset()
        # self.subtask_idx = 0 self.subtask_idx = self.get_next_subtask()
        # self.count = self.subtask.count
        # return self.get_observation()

    def get_next_subtask(self):
        if self.subtask_idx > self.n_subtasks:
            return None
        return self.control[self.subtask_idx, int(self.evaluate_condition())]

    def evaluate_condition(self):
        self.pred = self.conditions[self.subtask_idx] in self.objects.values()
        return self.pred

    def get_required_objects(self, _):
        yield from self.required_objects


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
