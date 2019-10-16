import itertools
import shutil
import numpy as np
import gym
from gym.utils import seeding
from collections import namedtuple

Obs = namedtuple("Obs", "sizes pos index")


class Env(gym.Env):
    def __init__(
        self,
        width,
        n_train: int,
        n_eval: int,
        speed: float,
        seed: int,
        time_limit: int,
        obs_type: str,
    ):
        self.time_limit = time_limit
        self.speed = speed
        self.n_eval = n_eval
        self.n_train = n_train
        self.sizes = None
        self.indices = None
        self.edges = None
        self.split = None
        self.width = width
        self.random, self.seed = seeding.np_random(seed)
        self.max_pictures = max(n_eval, n_train)
        self.observation_space = gym.spaces.Box(
            high=1, low=0, shape=(2, self.max_pictures, self.width)
        )
        self.action_space = gym.spaces.Discrete(self.width + 1)
        self.evaluating = False
        self.t = None
        self.i = None

    def step(self, action):
        next_picture = action >= self.width
        self.t += 1
        if self.t > self.time_limit:
            return self.get_observation(), -2 * self.width, True, {}
        if next_picture:
            if self.i < len(self.sizes) - 1:
                self.i += 1
            else:

                def compute_white_space():
                    left = 0
                    for right, picture in zip(self.edges, self.sizes):
                        yield right - left
                        left = right + picture
                    yield self.width - left

                white_space = list(compute_white_space())
                # max reward is 0
                return (
                    self.get_observation(),
                    (min(white_space) - max(white_space)),
                    True,
                    {},
                )
        else:
            edge = self.edges[self.i]
            desired_delta = action - edge
            delta = min(abs(desired_delta), self.speed) * (
                1 if desired_delta > 0 else -1
            )
            self.edges[self.i] = max(
                0, min(self.width - self.sizes[self.i], edge + delta)
            )
        return self.get_observation(), 0, False, {}

    def reset(self):
        self.t = 0
        self.i = 0
        self.indices = list(range(self.max_pictures))
        self.random.shuffle(self.indices)
        n_pictures = self.random.random_integers(1, self.n_train)
        self.split = self.random.randint(self.max_pictures - n_pictures)
        randoms = self.random.random(self.n_eval if self.evaluating else n_pictures)
        normalized = randoms * self.width / randoms.sum()
        cumsum = np.round(np.cumsum(normalized)).astype(int)
        z = np.roll(np.append(cumsum, 0), 1)
        self.sizes = z[1:] - z[:-1]
        self.sizes = self.sizes[self.sizes > 0]
        self.edges = [
            self.random.random_integers(0, self.width - size) for size in self.sizes
        ]
        self.random.shuffle(self.sizes)
        return self.get_observation()

    def new_position(self):
        return int(self.random.random() * self.width)

    def get_observation(self):
        obs = self.raw_observation()[self.indices]
        obs = np.array([[0, 0], [1, 0], [0, 1]])[obs].transpose((2, 0, 1))
        self.observation_space.contains(obs)
        return obs

    def raw_observation(self):
        zero = [[0] * self.width] * (self.max_pictures - len(self.edges))
        nonzero = [
            [0] * edge + ([1] if i == self.i else [2]) * size
            for i, (edge, size) in enumerate(zip(self.edges, self.sizes))
        ]
        nonzero = [row[: self.width] + [0] * (self.width - len(row)) for row in nonzero]
        obs = zero[: self.split] + nonzero + zero[self.split :]
        return np.array(obs)

    def pad(self, obs):
        if len(obs) == self.max_pictures:
            return obs
        return np.pad(obs, (0, self.max_pictures - len(obs)), constant_values=-1)

    def render(self, mode="human", pause=True):
        np.set_printoptions(
            threshold=self.width * self.max_pictures, linewidth=2 * self.width + 4
        )
        print(self.raw_observation())
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
            a, b = string.split()
            return float(a), int(b)
        except ValueError:
            return

    keyboard_control.run(Env(**args), action_fn=action_fn)
