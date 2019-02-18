import hsr
from hsr.env import Observation
from utils import concat_spaces, vectorize, space_shape


class HSREnv(hsr.env.HSREnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Sadly, ppo code really likes boxes, so had to concatenate things
        self.observation_space = concat_spaces(self.observation_space.spaces)

    def step(self, action):
        s, r, t, i = super().step(action)
        return vectorize(s), r, t, i

    def reset(self):
        return vectorize(super().reset())


class GoalsHSREnv(hsr.env.HSREnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        old_spaces = hsr.env.Observation(*self.observation_space.spaces)
        spaces = Observation(
            observation=old_spaces.observation, goal=old_spaces.goal)

        # subspace_sizes used for splitting concatenated tensor observations
        self._subspace_sizes = Observation(
            *[space_shape(space)[0] for space in spaces])
        for n in self.subspace_sizes:
            assert isinstance(n, int)

        # space of observation needs to exclude reward param
        import ipdb
        ipdb.set_trace()
        self.observation_space = concat_spaces(spaces)
        self.reward_params = self.achieved_goal()

    @property
    def subspace_sizes(self):
        return self._subspace_sizes

    def step(self, actions):
        s, r, t, i = super().step(actions)
        i.update(goal=self.goal)
        observation = Observation(observation=s.observation, goal=s.goal)
        return vectorize(observation), r, t, i

    def reset(self):
        o = super().reset()
        return vectorize(Observation(observation=o.observation, goal=o.goal))

    def new_goal(self):
        return self.goal


class MoveGripperEnv(HSREnv, hsr.env.MoveGripperEnv):
    pass


class GoalsMoveGripperEnv(GoalsHSREnv, hsr.env.MoveGripperEnv):
    pass