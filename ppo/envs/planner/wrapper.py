import gym
from collections import namedtuple

Actions = namedtuple("Actions", "searches actual")


class Wrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):



