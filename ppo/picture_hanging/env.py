import shutil

import gym
from gym.utils import seeding


class Env(gym.Env):
    def __init__(self, width, n_pictures, seed):
        self.n_pictures = n_pictures
        self.center = None
        self.sizes = None
        self.pictures = list(range(n_pictures))
        self.assigned_pictures = None
        self.width = width
        self.random, self.seed = seeding.np_random(seed)

    def step(self, center):
        picture, center = center
        self.center.append(center)
        self.assigned_pictures.append(picture)
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
            if tuple(self.assigned_pictures) == tuple(self.sizes):
                r = min(white_space) - max(white_space)  # max reward is 0
            else:
                r = -self.width

        return self.get_observation(), r, t, {}

    def reset(self):
        self.center = []
        self.assigned_pictures = []
        self.sizes = [
            self.random.rand() * self.width / self.n_pictures
            for _ in range(self.n_pictures)
        ]
        self.random.shuffle(self.pictures)
        return self.get_observation()

    def get_observation(self):
        return self.pictures, self.sizes

    def render(self, mode="human", pause=True):
        terminal_width = shutil.get_terminal_size((80, 20)).columns
        ratio = terminal_width / self.width
        right = 0
        for picture in self.sizes:
            print("=" * int(round(picture * ratio)))
        for center, picture in zip(self.center, self.sizes):
            left = center - picture / 2
            print(" " * int(round((left - right) * ratio)), end="")
            print("=" * int(round(picture * ratio)), end="")
            right = center + picture / 2
        print()
        if pause:
            input("pause")


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
