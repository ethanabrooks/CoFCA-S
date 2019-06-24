import ppo


class Wrapper(ppo.subtasks.wrappers.Wrapper):
    def wrap_observation(self, observation):
        obs = super().wrap_observation(observation)
        raise NotImplementedError
