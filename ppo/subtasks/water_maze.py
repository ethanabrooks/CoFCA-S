import gym


class WaterMaze(gym.Env):
    def __init__(self, time_limit, platform_size):
        self.platform_size = platform_size
        self.time_limit = time_limit

    def generator(self):
        for t in range(self.time_limit):
            yield

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode="human"):
        pass
