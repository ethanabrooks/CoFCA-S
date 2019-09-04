import abc

import gym


class Env(gym.Env):
    def __init__(self):
        super().__init__()
        self.active_subtask = None

    def step(self, action):
        r = 0
        t = False
        if self.active_subtask.done():
            self.active_subtask = self.next_subtask()
            if self.active_subtask is None:
                r = 1
                t = True
        return self.get_observation(), r, t, {}

    def reset(self):
        self.active_subtask = self.initial_subtask()
        return self.get_observation()

    def render(self, mode="human"):
        pass

    @abc.abstractmethod
    def next_subtask(self):
        pass

    @abc.abstractmethod
    def initial_subtask(self):
        pass

    @abc.abstractmethod
    def get_observation(self):
        pass
