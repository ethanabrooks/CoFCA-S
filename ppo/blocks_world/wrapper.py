from collections import namedtuple

import gym

Actions = namedtuple("Actions", "searches actual")

class Wrapper(gym.ObservationWrapper):
    def observation(self, observation):

