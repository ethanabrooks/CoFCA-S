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
        self,
        time_limit,
        platform_size,
        show_platform,
        movement_size,
        render_size=100,
        seed=0,
    ):
        self.show_platform = show_platform
        self.render_size = render_size
        self.movement_size = movement_size
        self.platform_size = platform_size
        self.time_limit = time_limit
        self.random, self.seed = seeding.np_random(seed)
        self.iterator = None
        self._render = None
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0, -1]), high=np.array([1, 1, 1, 1, 1, 0])
        )
        a = np.array([movement_size, movement_size, 1])
        self.action_space = gym.spaces.Box(-a, a)

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
        positions = [position]
        info = {}
        exploring = True
        for t in itertools.count():

            def render():
                array = np.zeros((self.render_size, self.render_size))

                def scale(x):
                    return x * self.render_size

                self.draw_points(
                    scale(platform_center),
                    value=-1,
                    array=array,
                    radius=scale(self.platform_size),
                )
                self.draw_points(
                    *map(scale, positions), value=1, array=array, radius=scale(0.05),
                )
                self.draw_points(
                    scale(position), value=2, array=array, radius=scale(0.1),
                )
                i = 15
                for row in array:
                    for cell in row:
                        j = int(
                            round(
                                (cell + array.min()) / (array.max() - array.min()) * i
                            )
                        )
                        code = str(i * 16 + j)
                        sys.stdout.write("\u001b[48;5;" + code + "m  ")
                    print("\u001b[0m")
                print("position:", position)

            self._render = render

            on_platform = (
                np.linalg.norm(position - platform_center) < self.platform_size
            )
            if exploring:
                reward = 0
                term = False
                if on_platform:
                    exploring = False
                    position = self.random.random(2)
            else:
                reward = on_platform - 1
                term = on_platform or t == self.time_limit
            if term:
                info.update(time=t, success=on_platform)
            obs = tuple(
                (
                    *position,
                    *(platform_center if self.show_platform else [0, 0]),
                    reward,
                    exploring,
                )
            )
            *movement, done_exploring = (
                yield obs,
                reward,
                term,
                info,
            )
            movement = np.array(movement)
            movement *= self.movement_size / np.linalg.norm(movement)
            position += movement
            position = np.clip(position, 0, 1)
            positions.append(position)

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
        parser.add_argument("--platform-size", default=0.1, type=float)
        parser.add_argument("--render-size", default=100)
        parser.add_argument("--movement-size", default=0.1, type=float)
        parser.add_argument("--show-platform", action="store_true")


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser()
    WaterMaze.add_arguments(PARSER)
    PARSER.add_argument("--time-limit", default=100)
    WaterMaze(**vars(PARSER.parse_args())).interact()
