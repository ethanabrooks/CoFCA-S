import itertools
from collections import namedtuple
import numpy as np

from gym import spaces

from gridworld_env import SubtasksGridWorld

Branch = namedtuple('Branch', 'condition true_path false_path')


class ControlFlowGridWorld(SubtasksGridWorld):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conditions = None
        self.control = None
        self.subtasks = None
        obs_space, subtasks_space = self.observation_space.spaces
        self.observation_space = spaces.Tuple([
            obs_space, subtasks_space,
            spaces.MultiBinary(self.n_subtasks**2)
        ])

    def render_task(self):
        def helper(i, indent):
            neg, pos = self.control[i]
            condition = self.conditions[i]

            def develop_branch(j, add_indent):
                new_indent = indent + add_indent
                try:
                    subtask = f'{j}:{self.subtasks[j]}'
                except IndexError:
                    return f'{new_indent}terminate'
                return f"{new_indent}{subtask}\n{helper(j, new_indent)}"

            if pos == neg:
                return f"{develop_branch(pos, '')}"
            else:
                return f'''\
{indent}if {condition}:{self.object_types[condition]}:
{develop_branch(pos, '    ')}
{indent}else:
{develop_branch(neg, '    ')}
'''

        print(helper(i=0, indent=''))

    def get_observation(self):
        obs, task = super().get_observation()
        return obs, task, self.control

    def task_generator(self):
        choices = self.np_random.choice(
            len(self.possible_subtasks), size=self.n_subtasks + 1)
        for subtask in self.possible_subtasks[choices]:
            yield self.Subtask(*subtask)

    def reset(self):
        o = super().reset()
        n = self.n_subtasks + 1

        def get_control():
            for i in range(n):
                yield self.np_random.randint(
                    i + 1,
                    n + (i > 0),  # prevent termination on first turn
                    size=2)

        self.control = np.array(list(get_control()))
        self.conditions = self.np_random.choice(len(self.object_types), size=n)
        self.subtask_idx = 0
        self.subtask_idx = self.get_next_subtask()
        self.task_count = self.subtask.count
        return o

    def get_next_subtask(self):
        object_type = self.conditions[self.subtask_idx]
        resolution = object_type in self.objects.values()
        return self.control[self.subtask_idx, int(resolution)]

    def get_required_objects(self, _):
        required_objects = list(super().get_required_objects(self.subtasks))
        yield from required_objects
        # for line in self.task:
        #     if isinstance(line, self.Branch):
        #         if line.condition not in required_objects:
        #             if self.np_random.rand() < .5:
        #                 yield line.condition


def main(seed, n_subtasks):
    kwargs = gridworld_env.get_args('4x4SubtasksGridWorld-v0')
    del kwargs['class_']
    del kwargs['max_episode_steps']
    kwargs.update(interactions=['pick-up', 'transform'], n_subtasks=n_subtasks)
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
