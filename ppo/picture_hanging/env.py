import itertools
import shutil
import numpy as np
import gym
from gym.utils import seeding
from collections import namedtuple

Obs = namedtuple("Obs", "sizes obs")


class Env(gym.Env):
    def __init__(
        self, width, n_train: int, n_eval: int, speed: float, seed: int, time_limit: int
    ):
        self.time_limit = time_limit
        self.speed = speed
        self.n_eval = n_eval
        self.n_train = n_train
        self.sizes = None
        self.centers = None
        self.width = width
        self.random, self.seed = seeding.np_random(seed)
        self.max_pictures = max(n_eval, n_train)
        box = gym.spaces.Box(low=0, high=self.width, shape=(self.max_pictures,))
        self.observation_space = gym.spaces.Dict(Obs(sizes=box, obs=box)._asdict())
        # self.action_space = gym.spaces.Discrete(self.width)
        self.action_space = gym.spaces.Dict(
            goal=gym.spaces.Box(low=0, high=self.width, shape=(1,)),
            next=gym.spaces.Discrete(2),
        )
        self.evaluating = False
        self.t = None

    def step(self, actions):
        goal, next_picture = actions
        self.t += 1
        if self.t > self.time_limit:
            return self.get_observation(), -self.width, True, {}
        if next_picture:
            if len(self.centers) == len(self.sizes):

                def compute_white_space():
                    left = 0
                    for center, picture in zip(self.centers, self.sizes):
                        right = center - picture / 2
                        yield right - left
                        left = center + picture / 2
                    yield self.width - left

                white_space = list(compute_white_space())
                # max reward is 0
                return (
                    self.get_observation(),
                    (min(white_space) - max(white_space)),
                    True,
                    {},
                )
            self.centers.append(0)
        self.centers[-1] = max(
            0, min(self.width, min(goal, self.centers[-1] + self.speed))
        )
        return self.get_observation(), 0, False, {}

    def reset(self):
        self.t = 0
        self.centers = []
        self.sizes = self.random.random(
            self.n_eval
            if self.evaluating
            else self.random.random_integers(2, self.n_train)
        )
        self.sizes = self.sizes * self.width / self.sizes.sum()
        self.random.shuffle(self.sizes)
        return self.get_observation()

    def get_observation(self):
        obs = Obs(sizes=self.pad(self.sizes), obs=self.pad(self.centers))._asdict()
        self.observation_space.contains(obs)
        return obs

    def pad(self, obs):
        if len(obs) == self.max_pictures:
            return obs
        return np.pad(obs, (0, self.max_pictures - len(obs)), constant_values=-1)

    def render(self, mode="human", pause=True):
        terminal_width = shutil.get_terminal_size((80, 20)).columns
        ratio = terminal_width / self.width
        right = 0
        for i, picture in enumerate(self.sizes):
            print(str(i) * int(round(picture * ratio)))
        print("placements")
        for i, (center, picture) in enumerate(zip(self.centers, self.sizes)):
            left = center - picture / 2
            print("-" * int(round(left * ratio)), end="")
            print(str(i) * int(round(picture * ratio)))
            right = center + picture / 2
        print("-" * int(round(self.width * ratio)))
        if pause:
            input("pause")

    def increment_curriculum(self):
        raise NotImplementedError

    def train(self):
        self.evaluating = False

    def evaluate(self):
        self.evaluating = True


if __name__ == "__main__":
    import argparse
    from rl_utils import hierarchical_parse_args, namedtuple
    from ppo import keyboard_control

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--width", default=100, type=int)
    parser.add_argument("--n-train", default=4, type=int)
    parser.add_argument("--n-eval", default=6, type=int)
    parser.add_argument("--speed", default=100, type=int)
    parser.add_argument("--time-limit", default=100, type=int)
    args = hierarchical_parse_args(parser)

    def action_fn(string):
        try:
            return float(string), 1
        except ValueError:
            return

    keyboard_control.run(Env(**args), action_fn=action_fn)
