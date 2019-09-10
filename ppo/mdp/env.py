from collections import namedtuple

import gym
import numpy as np
from gym.utils import seeding

Obs = namedtuple("Obs", "mdp values")
Last = namedtuple("last", "action reward terminal")


class Env(gym.Env):
    def __init__(self, n_states: int, seed: int, time_limit: int, delayed_reward: bool):
        self.delayed_reward = delayed_reward
        self.time_limit = time_limit
        self.random, self.seed = seeding.np_random(seed)
        self.n_states = n_states
        self.mdp = None
        self.values = None
        self.current = None
        self.correct = None
        self.last = None
        self.t = None
        self.action_space = gym.spaces.Discrete(n_states)
        self.observation_space = gym.spaces.Dict(
            Obs(
                mdp=gym.spaces.MultiDiscrete(2 * np.ones([n_states, n_states])),
                values=gym.spaces.Box(low=0, high=1, shape=(n_states,)),
            )._asdict()
        )

    def step(self, action: int):
        action = int(action)
        self.t += 1
        t = np.all(self.mdp[self.current] == 0) or self.t == self.time_limit
        correct = self.q_values()[action] == self.q_values().max()
        if not correct:
            self.correct = False
        if self.delayed_reward:
            r = self.correct if t else 0
        else:
            r = float(correct)
        if not self.mdp[self.current, action]:
            # if transition is possible
            self.current = action
        self.last = Last(action=action, reward=r, terminal=t)
        return self.get_observation(), r, t, {}

    def q_values(self):
        return (self.values * self.mdp)[self.current]

    def reset(self):
        shape = [self.n_states, self.n_states]
        self.mdp = self.random.randint(0, 2, shape)
        self.values = self.random.rand(*shape)
        self.correct = True
        self.current = 0
        self.t = 0
        self.last = None
        return self.get_observation()

    def render(self, mode="human", pause=True):
        for k, x in self.get_observation().items():
            print(k)
            print(x)
        if self.last:
            print(self.last)
        print("current state:", self.current)
        if pause:
            input("pause")

    def get_observation(self):
        q_values = self.q_values()
        obs = Obs(mdp=self.mdp, values=self.values[self.current])._asdict()
        assert self.observation_space.contains(obs)
        return obs  # TODO: just send q_values


if __name__ == "__main__":
    import argparse
    from rl_utils import hierarchical_parse_args
    from ppo import keyboard_control

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--n-states", default=4, type=int)
    parser.add_argument("--time-limit", default=8, type=int)
    args = hierarchical_parse_args(parser)

    def action_fn(string):
        try:
            action = int(string)
            assert action < args["n_states"]
            return action
        except (ValueError, AssertionError):
            return

    keyboard_control.run(Env(**args), action_fn=action_fn)
