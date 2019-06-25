import itertools
from collections import namedtuple

from gridworld_env import SubtasksGridWorld

Branch = namedtuple('Branch', 'condition true_path false_path')


class ControlFlowGridWorld(SubtasksGridWorld):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subtasks = None
        world = self

        class _Branch(Branch):
            # noinspection PyMethodParameters
            def __str__(self):
                return f'''\
if {world.object_types[self.condition]}:
    {self.true_path}
else:
    {self.false_path}
'''

        self.Branch = _Branch

    def subtask_generator(self):
        max_value = 0
        choices = self.np_random.choice(
            len(self.possible_subtasks), size=self.n_subtasks)
        self.subtasks = [
            self.Subtask(*x) for x in self.possible_subtasks[choices]
        ]
        for i in itertools.count():
            branch1 = self.np_random.random_integers(i, max_value + 1)
            max_value = max(max_value, branch1)
            branch2 = self.np_random.random_integers(i, max_value + 1)
            max_value = max(max_value, branch2)
            if branch1 == branch2:
                yield self.Subtask(self.possible_subtasks[branch1])
            else:
                yield self.Branch(
                    condition=self.np_random.choice(len(self.object_types)),
                    true_path=self.subtasks[branch1],
                    false_path=self.subtasks[branch1],
                )
            if max_value == self.n_subtasks:
                raise StopIteration

    def get_next_subtask(self):
        if isinstance(self.subtask, self.Branch):
            resolution = self.evaluate_condition(self.subtask.condition)
            return [self.subtask.false_path,
                    self.subtask.true_path][resolution]
        return self.subtask

    def required_objects(self, _):
        yield from super().required_objects(self.subtasks)

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
