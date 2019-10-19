import itertools
import shutil
import numpy as np
import gym
from gym.utils import seeding
from collections import namedtuple

Obs = namedtuple("Obs", "sizes obs")


class Env(gym.Env):
    def __init__(
        self,
        width,
        min_train: int,
        max_train: int,
        n_eval: int,
        speed: float,
        seed: int,
        time_limit: int,
        include_sizes: bool,
    ):
        self.include_sizes = include_sizes
        self.time_limit = time_limit
        self.speed = speed
        self.n_eval = n_eval
        self.min_train = min_train
        self.max_train = max_train
        self.sizes = None
        self.edges = None
        self.new_picture = None
        self.observation_iterator = None
        self.width = width
        self.random, self.seed = seeding.np_random(seed)
        self.max_pictures = max(n_eval, max_train)
        self.observation_space = gym.spaces.MultiDiscrete(2 * np.ones(self.width + 3))
        if include_sizes:
            self.observation_space = gym.spaces.Dict(
                Obs(
                    sizes=gym.spaces.MultiDiscrete(
                        self.width * np.ones(self.max_pictures)
                    ),
                    obs=self.observation_space,
                )._asdict()
            )
        self.action_space = gym.spaces.Discrete(2 * self.width + 1)
        self.evaluating = False
        self.t = None
        self.eye = np.eye(self.width + 1)

    def step(self, action):
        next_picture = action >= self.width
        self.new_picture = next_picture
        self.t += 1
        if self.t > self.time_limit:
            return self.get_observation(), -2 * self.width, True, {}
        if next_picture:
            if len(self.edges) < len(self.sizes):
                self.edges += [self.new_position()]
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
            edge = self.edges[-1]
            desired_delta = action - edge
            delta = min(abs(desired_delta), self.speed) * (
                1 if desired_delta > 0 else -1
            )
            self.edges[-1] = max(0, min(self.width - self.sizes[-1], edge + delta))
        return self.get_observation(), 0, False, {}

    def reset(self):
        self.t = 0
        self.new_picture = True
        n_pictures = self.random.random_integers(self.min_train, self.max_train)
        randoms = self.random.random(self.n_eval if self.evaluating else n_pictures)
        normalized = randoms / randoms.sum() * self.width
        cumsum = np.round(np.cumsum(normalized)).astype(int)
        z = np.roll(np.append(cumsum, 0), 1)
        self.sizes = z[1:] - z[:-1]
        self.sizes = self.sizes[self.sizes > 0]
        # gap = self.random.randint(0, self.sizes.min())
        # self.sizes -= gap
        self.edges = [self.new_position()]
        self.observation_iterator = self.observation_generator()
        return self.get_observation()

    def new_position(self):
        return int(self.random.random() * self.width)

    def observation_generator(self):
        for size in self.sizes:
            yield list(self.eye[size]) + [0, self.new_picture]
        while True:
            yield list(self.eye[int(self.edges[-1])]) + [1, self.new_picture]

    def get_observation(self):
        obs = next(self.observation_iterator)
        if self.include_sizes:
            obs = Obs(sizes=self.pad(self.sizes), obs=obs)._asdict()
        self.observation_space.contains(obs)
        return obs

    def pad(self, obs):
        if len(obs) == self.max_pictures:
            return obs
        return np.pad(obs, (0, self.max_pictures - len(obs)), constant_values=-1)

    def render(self, mode="human", pause=True):
        print("sizes", self.sizes)
        # np.set_printoptions(
        # threshold=self.width * self.max_pictures, linewidth=4 * (self.width + 1)
        # )
        state2d = [
            [0] * edge + [1] * size
            for i, (edge, size) in enumerate(zip(self.edges, self.sizes))
        ]
        state2d = np.array(
            [row[: self.width] + [0] * (self.width - len(row)) for row in state2d]
        )
        string = np.array([["-"], ["#"]])[state2d, 0]
        for row in string:
            print(*row)
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
    parser.add_argument("--width", default=3, type=int)
    parser.add_argument("--min-train", default=2, type=int)
    parser.add_argument("--max-train", default=2, type=int)
    parser.add_argument("--n-eval", default=6, type=int)
    parser.add_argument("--speed", default=3, type=int)
    parser.add_argument("--time-limit", default=100, type=int)
    args = hierarchical_parse_args(parser)

    def action_fn(string):
        try:
            return int(string)
        except ValueError:
            return

    keyboard_control.run(Env(**args), action_fn=action_fn)
