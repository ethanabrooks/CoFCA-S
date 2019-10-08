import itertools
import shutil
import numpy as np
import gym
from gym.utils import seeding


class Env(gym.Env):
    def __init__(self, width, n_train, n_eval, single_step, seed):
        self.single_step = single_step
        self.n_train = n_train
        self.n_eval = n_eval
        self.max_pictures = max(n_train, n_eval)
        self.centers = None
        self.sizes = None
        self.n_pictures = n_train
        self.assigned_pictures = None
        self.width = width
        self.random, self.seed = seeding.np_random(seed)
        self.observation_space = gym.spaces.Box(low=0, high=self.width, shape=(n_eval,))
        # self.action_space = gym.spaces.Discrete(self.width)
        if single_step:
            self.action_space = gym.spaces.Box(
                low=0, high=self.width, shape=(self.max_pictures,)
            )
        else:
            self.action_space = gym.spaces.Box(low=0, high=self.width, shape=(1,))
        self.train_sizes = self.width * self.random.random(n_train)
        self.eval_sizes = np.array(
            list(itertools.islice(itertools.cycle(self.train_sizes), n_eval))
        )
        self.evaluating = False

    def step(self, center):
        if self.single_step:
            self.centers = center
        else:
            self.centers.append(center)
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
        if t:
            i.update(reward_plus_n_picturs=len(self.sizes) + r)
        return self.get_observation(), r, t, i

    def reset(self):
        self.centers = []
        self.sizes = self.train_sizes
        self.random.shuffle(self.sizes)
        return self.get_observation()

    def get_observation(self):
        obs = self.sizes
        if len(self.sizes) < self.max_pictures:
            obs = np.pad(
                self.sizes, (0, self.max_pictures - len(self.sizes)), constant_values=-1
            )
        self.observation_space.contains(obs)
        return obs

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
        self.n_pictures = min(self.n_pictures + 1, self.max_pictures)
        self.reset()

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
