from gym import spaces

import ppo.subtasks


class Wrapper(ppo.subtasks.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space.spaces.update(
            g=spaces.Discrete(len(env.observation_space.spaces["lines"].nvec))
        )
