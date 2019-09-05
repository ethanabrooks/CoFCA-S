import abc
import gym


class Env(gym.Env):
    def __init__(self):
        super().__init__()
        self.active = None

    def step(self, action):
        r = -0.1
        t = False
        if self.done():
            self.active = self.next()
            if self.active is None:
                r = 1
                t = True
        return self.get_observation(), r, t, {}

    def reset(self):
        self.active = self.initial()
        return self.get_observation()

    def render(self, mode="human"):
        pass

    @abc.abstractmethod
    def next(self):
        pass

    @abc.abstractmethod
    def initial(self):
        pass

    @abc.abstractmethod
    def get_observation(self):
        pass

    @abc.abstractmethod
    def done(self):
        pass
