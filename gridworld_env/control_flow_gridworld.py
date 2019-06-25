import itertools
from collections import namedtuple

from gym import spaces

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
    {self.true_path}: {world.subtasks[self.true_path]}
else:
    {self.false_path}: {world.subtasks[self.false_path]}
'''

        self.Branch = _Branch
        obs_space, subtasks_space = self.observation_space.spaces
        self.observation_space = spaces.Tuple([
            obs_space, subtasks_space,
            spaces.MultiBinary(self.n_subtasks**2)
        ])

        class NonBranch(self.Subtask):
            def __new__(cls, i):
                cls.i = i
                return super().__new__(cls, *world.subtasks[cls.i])

            def __str__(self):
                return f'{self.i}: {super().__str__()}'

        self.NonBranch = NonBranch

    @property
    def subtask(self):
        try:
            return self.subtasks[self.idx]
        except IndexError:
            return None

    def get_observation(self):
        obs, task = super().get_observation()
        return obs, task, self.subtasks

    def task_generator(self):
        choices = self.np_random.choice(
            len(self.possible_subtasks), size=self.n_subtasks)
        self.subtasks = [
            self.Subtask(*x) for x in self.possible_subtasks[choices]
        ]
        min_value = 0
        max_value = 1
        while True:

            def sample():
                return self.np_random.randint(
                    min_value, min(max_value, 1 + self.n_subtasks))

            branch1 = sample()
            max_value = max(max_value, branch1 + 2)
            branch2 = sample()
            max_value = max(max_value, branch2 + 2)
            min_value += 1  # no looping back to past subtasks
            max_value = max(max_value, min_value + 1)

            if len(self.subtasks) in [branch1, branch2]:
                raise StopIteration

            if branch1 == branch2:
                yield self.NonBranch(branch1)
            else:
                yield self.Branch(
                    condition=self.np_random.choice(len(self.object_types)),
                    true_path=branch1,
                    false_path=branch2,
                )

    def get_next_subtask(self):
        import ipdb
        ipdb.set_trace()
        if isinstance(self.subtask, self.Branch):
            resolution = self.evaluate_condition(self.subtask.condition)
            return [self.subtask.false_path,
                    self.subtask.true_path][resolution]
        return super().get_next_subtask()

    def get_required_objects(self, _):
        required_objects = list(super().get_required_objects(self.subtasks))
        yield from required_objects
        # for line in self.task:
        #     if isinstance(line, self.Branch):
        #         if line.condition not in required_objects:
        #             if self.np_random.rand() < .5:
        #                 yield line.condition

    def evaluate_condition(self, object_type):
        return object_type in self.objects.values()


def main(seed):
    kwargs = gridworld_env.get_args('4x4SubtasksGridWorld-v0')
    del kwargs['class_']
    del kwargs['max_episode_steps']
    kwargs.update(interactions=['pick-up', 'transform'])
    env = ControlFlowGridWorld(**kwargs, evaluation=False, eval_subtasks=[])
    actions = 'wsadeq'
    gridworld_env.keyboard_control.run(env, actions=actions, seed=seed)


if __name__ == '__main__':
    import argparse
    import gridworld_env.keyboard_control
    parser = argparse.ArgumentParser()
    parser.add_argument('seed', type=int)
    main(**vars(parser.parse_args()))
