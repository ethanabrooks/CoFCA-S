import itertools

import gym
import numpy as np
from gym.utils import seeding


class Bandit(gym.Env):
    def __init__(
        self, n: int, time_limit: int, explore_limit: int, std: float, seed=0,
    ):
        assert explore_limit < time_limit
        self.n = n
        self.std = std
        self.explore_limit = explore_limit
        self.time_limit = time_limit
        self.random, self.seed = seeding.np_random(seed)
        self.iterator = None
        self._render = None
        self.observation_space = gym.spaces.Box(
            low=np.array([-1, -1, 0]), high=np.array([1, n, 1])
        )
        self.action_space = gym.spaces.Discrete(n)

    def generator(self):
        statistics = np.zeros(self.n)
        statistics[int(self.random.randint(self.n))] = 1
        best = statistics.max()
        reward = -1
        action = -1
        info = {}
        cumulative_rewards = np.zeros(self.n)
        choices = np.zeros(self.n)
        for t in itertools.count():
            exploring = t <= self.explore_limit

            def render():
                for i, stat in enumerate(statistics):
                    print(i, stat)
                print("action:", action)
                print("reward", reward)
                print("exploring", exploring)

            self._render = render

            obs = (reward, action, exploring)
            term = t >= self.time_limit

            action = yield obs, 0 if exploring else reward, term, info
            reward = self.random.normal(statistics[action], self.std)

            # infos
            cumulative_rewards[action] += reward
            choices[action] += 1
            if not exploring:
                known_statistics = cumulative_rewards / t
                info = dict(
                    regret=best - reward,
                    choose_best=action == statistics.argmax(),
                    choose_best_known=action == known_statistics.argmax(),
                )

    def step(self, action):
        return self.iterator.send(action)

    def reset(self):
        self.iterator = self.generator()
        s, _, _, _, = next(self.iterator)
        return s

    def render(self, mode="human", pause=True):
        self._render()
        if pause:
            input()

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--n", type=int)
        parser.add_argument("--time-limit", type=int)
        parser.add_argument("--explore-limit", type=int)
        parser.add_argument("--std", type=float)


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser()
    Bandit.add_arguments(PARSER)
    # Bandit(**vars(PARSER.parse_args())).interact()
