import itertools
import sys
import time

import gym
import numpy as np
import skimage.draw
import skimage.transform
from gym.utils import seeding


class WaterMaze(gym.Env):
    def __init__(
        self, time_limit, platform_size, render_size=100, seed=0,
    ):
        self.render_size = render_size
        self.platform_size = platform_size
        self.time_limit = time_limit
        self.random, self.seed = seeding.np_random(seed)
        self.iterator = None
        self._render = None
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, -1]), high=np.array([1, 1, 0])
        )
        self.action_space = gym.spaces.Box(-np.ones(3), np.ones(3))

    @staticmethod
    def draw_points(*points, array, value, **kwargs):
        for point in points:
            rr, cc = skimage.draw.circle(*point, **kwargs)
            i, j = array.shape

            condition = np.logical_and(
                np.logical_and(0 <= rr, rr < i), np.logical_and(0 <= cc, cc < j),
            )
            rr = rr[condition]
            cc = cc[condition]
            array[rr, cc] = value
        return array

    def generator(self):
        platform_center = self.random.random(2)
        position = self.random.random(2)
        info = {}
        for t in itertools.count():

            def render():
                array = np.zeros((self.render_size, self.render_size))

                def scale(x):
                    return x * self.render_size

                self.draw_points(
                    scale(position), value=1, array=array, radius=scale(0.1),
                )
                self.draw_points(
                    scale(platform_center),
                    value=-1,
                    array=array,
                    radius=scale(self.platform_size),
                )
                i = 15
                for row in array:
                    for cell in row:
                        j = int(round((cell + 1) / 2 * i))
                        code = str(i * 16 + j)
                        sys.stdout.write(u"\u001b[48;5;" + code + "m  ")
                    print(u"\u001b[0m")
                print("position:", position)

            self._render = render

            on_platform = (
                np.linalg.norm(position - platform_center) < self.platform_size
            )
            reward = on_platform - 1
            term = on_platform
            if term:
                info.update(time=t)
            *movement, done_exploring = yield (
                tuple((*position, reward)),
                reward,
                term,
                info,
            )
            position += np.array(movement)
            position = np.clip(position, 0, 1)

    def step(self, action):
        return self.iterator.send(action)

    def reset(self):
        self.iterator = self.generator()
        s, _, _, _ = next(self.iterator)
        return s

    def render(self, mode="human", pause=True):
        self._render()
        if pause:
            input()

    def train(self):
        pass

    def interact(self):
        actions = dict(w=(-0.1, 0), s=(0.1, 0), a=(0, -0.1), d=(0, 0.1))
        self.reset()
        while True:
            self.render(pause=False)
            action = None
            while action not in actions:
                action = input("act:")

            s, r, t, i = self.step(actions[action])
            print("reward", r)
            if t:
                self.render()
                print("resetting")
                time.sleep(0.5)
                self.reset()
                print()

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--platform-size", default=0.1)
        parser.add_argument("--render-size", default=50)


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser()
    WaterMaze.add_arguments(PARSER)
    PARSER.add_argument("--time-limit", default=100)
    WaterMaze(**vars(PARSER.parse_args())).interact()
