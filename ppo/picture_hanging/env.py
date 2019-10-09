import itertools
import shutil
import numpy as np
import gym
from gym.utils import seeding


class Env(gym.Env):
    def __init__(self, width, n_train: int, n_eval: int, single_step, seed):
        self.n_eval = n_eval
        self.n_train = n_train
        self.single_step = single_step
        self.sizes = None
        self.centers = None
        self.width = width
        self.random, self.seed = seeding.np_random(seed)
        self.max_pictures = max(n_eval, n_train)
        self.observation_space = gym.spaces.Box(
            low=-1, high=np.inf, shape=(self.max_pictures + 1,)
        )
        # self.action_space = gym.spaces.Discrete(self.width)
        if single_step:
            self.action_space = gym.spaces.Box(
                low=0, high=self.width, shape=(self.max_pictures,)
            )
        else:
            self.action_space = gym.spaces.Box(low=0, high=self.width, shape=(1,))
        self.evaluating = False

    def step(self, center):
        if self.single_step:
            self.centers = np.maximum(center, 0)
        else:
            self.centers.append(max(center, 0))
        t = False
        r = 0
        if len(self.centers) == len(self.sizes):
            t = True

            def compute_white_space():
                left = 0
                for center, picture in zip(self.centers, self.sizes):
                    right = center - picture / 2
                    yield right - left
                    left = center + picture / 2
                yield self.width - left

            white_space = list(compute_white_space())
            r = min(white_space) - max(white_space)  # max reward is 0

        i = dict(n_pictures=len(self.sizes))
        obs = self.pad(np.concatenate([[1], self.centers]))
        return obs, r, t, i

    def reset(self):
        self.centers = []
        self.sizes = self.random.random(
            self.n_eval
            if self.evaluating
            else self.random.random_integers(2, self.n_train)
        )
        self.sizes = self.sizes * self.width / self.sizes.sum()
        self.random.shuffle(self.sizes)
        return self.pad(np.concatenate([[0], self.sizes]))

    def pad(self, obs):
        obs_size = self.observation_space.shape[0]
        if obs.shape[0] < obs_size:
            obs = np.pad(obs, (0, obs_size - obs.size), constant_values=-1)
        assert self.observation_space.contains(obs)
        return obs

    # def get_observation(self):
    #     obs = self.sizes
    #     if len(self.sizes) < self.max_pictures:
    #         obs = np.pad(
    #             self.sizes, (0, self.max_pictures - len(self.sizes)), constant_values=-1
    #         )
    #     self.observation_space.contains(obs)
    #     return obs

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
    from rl_utils import hierarchical_parse_args
    from ppo import keyboard_control

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--width", default=4, type=int)
    parser.add_argument("--n-actions", default=4, type=int)
    args = hierarchical_parse_args(parser)

    def action_fn(string):
        try:
            return float(string)
        except ValueError:
            return

    keyboard_control.run(Env(**args), action_fn=action_fn)
