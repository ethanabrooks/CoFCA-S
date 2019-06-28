from collections import namedtuple

from gym import spaces
import numpy as np

from gridworld_env import SubtasksGridWorld

Obs = namedtuple('Obs', 'base subtasks conditions control')


class ControlFlowGridWorld(SubtasksGridWorld):
    def __init__(self, *args, n_subtasks, force_branching=False, **kwargs):
        super().__init__(*args, n_subtasks=n_subtasks, **kwargs)
        self.pred = None
        self.force_branching = force_branching
        if force_branching:
            assert n_subtasks % 2 == 0

        self.conditions = None
        self.control = None
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
                return f'''\
{indent}if {self.object_types[condition]}:
{develop_branch(pos, '    ')}
{indent}else:
{develop_branch(neg, '    ')}
'''

        print(helper(i=0, indent=''))

    def get_observation(self):
        obs, task = super().get_observation()
        return Obs(base=obs, subtasks=task, control=self.control, conditions=self.conditions)

    def subtasks_generator(self):
        choices = self.np_random.choice(len(self.possible_subtasks), size=self.n_subtasks + 1)
        for subtask in self.possible_subtasks[choices]:
            yield self.Subtask(*subtask)

    def reset(self):
        o = super().reset()
        n = self.n_subtasks + 1

        def get_control():
            for i in range(self.n_subtasks + 1):
                j = 2 * i
                if self.force_branching or self.np_random.rand() < .7:
                    yield j, j + 1
                else:
                    yield j, j

        self.control = 1 + np.minimum(np.array(list(get_control())), self.n_subtasks)
        self.conditions = self.np_random.choice(len(self.object_types), size=n)
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
        required_objects = list(super().get_required_objects(self.subtasks))
        yield from required_objects
        # for condition in self.conditions:
        # if condition not in required_objects:
        # if self.np_random.rand() < .5:
        # yield condition


def main(seed, n_subtasks):
    kwargs = gridworld_env.get_args('4x4SubtasksGridWorld-v0')
    del kwargs['class_']
    del kwargs['max_episode_steps']
    kwargs.update(interactions=['pick-up', 'transform'], n_subtasks=n_subtasks, max_task_count=1)
    env = ControlFlowGridWorld(**kwargs, evaluation=False, eval_subtasks=[])
    actions = 'wsadeq'
    gridworld_env.keyboard_control.run(env, actions=actions, seed=seed)


if __name__ == '__main__':
    import argparse
    import gridworld_env.keyboard_control

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int)
    parser.add_argument('--n-subtasks', type=int)
    main(**vars(parser.parse_args()))
