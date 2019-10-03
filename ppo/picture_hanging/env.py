import shutil
import numpy as np
import gym
from gym.utils import seeding


class Env(gym.Env):
    def __init__(self, width, min_pictures, max_pictures, seed):
        self.min_pictures = min_pictures
        self.max_pictures = max_pictures
        self.center = None
        self.sizes = None
        self.n_pictures = min_pictures
        self.assigned_pictures = None
        self.width = width
        self.random, self.seed = seeding.np_random(seed)
        self.observation_space = gym.spaces.Box(
            low=0, high=self.width, shape=(max_pictures,)
        )
        # self.action_space = gym.spaces.Discrete(self.width)
        self.action_space = gym.spaces.Box(low=0, high=self.width, shape=(1,))

    def step(self, center):
        self.center.append(center)
        t = False
        r = 0
        if len(self.center) == len(self.sizes):
            t = True

            def compute_white_space():
                left = 0
                for center, picture in zip(self.center, self.sizes):
                    right = center - picture / 2
                    yield right - left
                    left = center + picture / 2
                yield self.width - left

            white_space = list(compute_white_space())
            r = min(white_space) - max(white_space)  # max reward is 0

        return self.get_observation(), r, t, {}

    def reset(self):
        self.center = []
        self.assigned_pictures = []

        def sizes():
            width = self.width
            for _ in range(self.n_pictures):
                picture_width = self.random.rand() * width
                yield picture_width
                width -= picture_width

        self.sizes = list(sizes())
        self.random.shuffle(self.sizes)
        return self.get_observation()

    def get_observation(self):
        obs = np.pad(self.sizes, (0, self.max_pictures - self.n_pictures))
        self.observation_space.contains(obs)
        return obs

    def render(self, mode="human", pause=True):
        terminal_width = shutil.get_terminal_size((80, 20)).columns
        ratio = terminal_width / self.width
        right = 0
        for i, picture in enumerate(self.sizes):
            print(str(i) * int(round(picture * ratio)))
        print("placements")
        for i, (center, picture) in enumerate(zip(self.center, self.sizes)):
            left = center - picture / 2
            print("-" * int(round(left * ratio)), end="")
            print(str(i) * int(round(picture * ratio)))
            right = center + picture / 2
        print("-" * int(round(self.width * ratio)))
        if pause:
            input("pause")

    def increment_curriculum(self):
        return  # TODO
        self.n_pictures = min(self.n_pictures + 1, self.max_pictures)
        self.reset()


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
