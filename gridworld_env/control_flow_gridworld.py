from collections import namedtuple
from dataclasses import dataclass

from gridworld_env import SubtasksGridWorld


class ControlFlowGridWorld(SubtasksGridWorld):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _self = self

        @dataclass
        class Branch:
            condition: int
            true_path: _self.Subtask
            false_path: _self.Subtask

            def __str__(branch):
                return f'''\
if {self.object_types[branch.condition]}:
    {branch.true_path}
else:
    {branch.false_path}
'''

        self.Branch = Branch

    def subtask_generator(self):
        while True:
            subtask1, subtask2 = self.np_random.choice(
                len(self.possible_subtasks), size=2)
            yield self.Branch(
                condition=(self.np_random.choice(len(self.object_types))),
                false_path=self.Subtask(*self.possible_subtasks[subtask1]),
                true_path=self.Subtask(*self.possible_subtasks[subtask2]))

    def get_next_subtask(self):
        next_line = next(self.task_iter)
        if isinstance(next_line, self.Branch):
            resolution = self.evaluate_condition(next_line.condition)
            return [next_line.false_path, next_line.true_path][resolution]
        return next_line

    def get_required_objects(self, branch):
        if self.np_random.binomial(1, .5):
            yield branch.condition
        yield from super().get_required_objects(branch.true_path)
        yield from super().get_required_objects(branch.false_path)

    def evaluate_condition(self, object_type):
        return object_type in self.objects.values()


if __name__ == '__main__':
    import gridworld_env.keyboard_control

    kwargs = gridworld_env.get_args('4x4SubtasksGridWorld-v0')
    del kwargs['class_']
    del kwargs['max_episode_steps']
    env = ControlFlowGridWorld(**kwargs, evaluation=False, eval_subtasks=[])
    actions = 'wsadeq'
    gridworld_env.keyboard_control.run(env, actions=actions)
