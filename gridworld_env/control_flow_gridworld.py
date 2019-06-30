from collections import Counter, namedtuple

from gym import spaces
import numpy as np

from gridworld_env import SubtasksGridWorld

Obs = namedtuple("Obs", "base subtask subtasks conditions control next_subtask pred")


class ControlFlowGridWorld(SubtasksGridWorld):
    def __init__(self, *args, n_subtasks, force_branching=False, **kwargs):
        super().__init__(*args, n_subtasks=n_subtasks, **kwargs)
        self.passing_objects = None
        self.failing_objects = None
        self.pred = None
        self.force_branching = force_branching
        if force_branching:
            assert n_subtasks % 2 == 0

        self.conditions = None
        self.control = None
        self.required_objects = None
        obs_space, subtasks_space = self.observation_space.spaces
        self.observation_space = spaces.Tuple(
            Obs(
                base=obs_space,
                subtasks=spaces.MultiDiscrete(
                    np.tile(subtasks_space.nvec[:1], [self.n_subtasks + 1, 1])),
                conditions=spaces.MultiDiscrete(
                    np.array([len(self.object_types)]).repeat(self.n_subtasks + 1)),
                control=spaces.MultiDiscrete(
                    np.tile(
                        np.array([[self.n_subtasks + 1]]),
                        [
                            self.n_subtasks + 1,
                            2  # binary conditions
                        ]))))

    def set_pred(self):
        try:
            self.pred = self.evaluate_condition()
        except IndexError:
            pass

    def render_current_subtask(self):
        if self.subtask_idx == 0:
            print('none')
        else:
            super().render_current_subtask()

    def render_task(self):
        def helper(i, indent):
            neg, pos = self.control[i]
            condition = self.conditions[i]

            def develop_branch(j, add_indent):
                new_indent = indent + add_indent
                if j == 0:
                    subtask = f''
                else:
                    try:
                        subtask = f'{j}:{self.subtasks[j]}'
                    except IndexError:
                        return f'{new_indent}terminate'
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
        choices = self.np_random.choice(len(self.possible_subtasks), size=self.n_subtasks + 1)
        subtasks = [self.Subtask(*self.possible_subtasks[i]) for i in choices]
        i = 0
        encountered_passing = []
        encountered_failing = []
        encountered_subtasks = []
        encountered_idxs = []
        while True:
            passing = self.conditions[i] in self.required_objects
            branching = self.control[i, 0] != self.control[i, 1]
            encountered_passing += [self.conditions[i] if branching and passing else None]
            encountered_failing += [self.conditions[i] if branching and not passing else None]
            i = self.control[i, int(passing)]
            if i > self.n_subtasks:
                break
            encountered_subtasks += [subtasks[i]]
            encountered_idxs += [i]

        self.required_objects = list(set(o for o in encountered_passing if o is not None))
        available = [x for x in self.required_objects]

        print(self.object_types)
        print('encountered_failing', encountered_failing)
        print('encountered_idxs', encountered_idxs)
        for i, subtask in enumerate(subtasks):
            if i in encountered_idxs:
                obj = subtask.object
                to_be_removed = subtask.interaction in {1, 2}
                print('available', available)
                if obj not in available:
                    print('subtask', subtask)
                    print('encountered_failing[:i + 1]', encountered_failing[:i + 1])
                    if (not to_be_removed and obj in encountered_failing) or (
                            to_be_removed and obj in encountered_failing[:i + 1]):
                        # choose a different object
                        obj = self.np_random.choice(self.required_objects)
                        subtasks[i] = subtask._replace(object=obj)

                    # add object to map
                    self.required_objects += [obj]
                    available += [obj]
                    print('required', self.required_objects)
                    print('available', available)

                if to_be_removed:
                    available.remove(obj)
                    print('available', available)

        yield from subtasks

    def reset(self):
        n = self.n_subtasks + 1

        def get_control():
            for i in range(n):
                j = 2 * i
                if self.force_branching or self.np_random.rand() < .7:
                    yield j, j + 1
                else:
                    yield j, j

        self.control = 1 + np.minimum(np.array(list(get_control())), self.n_subtasks)

        object_types = np.arange(len(self.object_types))
        self.np_random.shuffle(object_types)
        object_types = object_types.reshape(-1, 2)  # TODO: what if not % 2?
        conditions_idxs = self.np_random.choice(len(object_types), size=n)
        branching = (self.control[:, 0] != self.control[:, 1])
        passing = self.np_random.choice(2, size=n)
        self.conditions = object_types[conditions_idxs, passing]
        self.failing_conditions = self.conditions[branching * (1 - passing).astype(bool)]
        self.required_objects = list(object_types[:, 1])

        o = super().reset()
        self.subtask_idx = 0
        self.count = None
        self.iterate = True
        self.next_subtask = True
        self.pred = self.evaluate_condition()
        return o._replace(conditions=self.conditions, control=self.control)

    def get_next_subtask(self):
        if self.subtask_idx > self.n_subtasks:
            return None
        resolution = self.evaluate_condition()
        return self.control[self.subtask_idx, int(resolution)]

    def evaluate_condition(self):
        object_type = self.conditions[self.subtask_idx]
        return object_type in self.objects.values()

    def get_required_objects(self, _):
        yield from self.required_objects


def main(seed, n_subtasks):
    kwargs = gridworld_env.get_args('4x4SubtasksGridWorld-v0')
    del kwargs['class_']
    del kwargs['max_episode_steps']
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
