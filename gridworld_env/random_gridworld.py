#! /usr/bin/env python
from typing import Dict

from gym import spaces
import numpy as np

from gridworld_env.gridworld import GridWorld


class RandomGridWorld(GridWorld):
    def __init__(self, random: Dict[str, int] = None, *args, **kwargs):
        self.random = random
        self.random_states = None
        self.possible_choices = None
        super().__init__(*args, **kwargs)
        self.possible_choices = np.ravel_multi_index(
            np.where(np.logical_not(np.isin(self.desc, self.blocked))),
            dims=self.desc.shape,
        )
        assert self.possible_choices.size
        self.observation_space = spaces.Tuple(
            [self.observation_space] * (1 + len(random))
        )

    def append_randoms(self, state):
        return (state, *map(int, self.random_states))

    def set_randoms(self):
        n_choices = sum(self.random.values())
        possible_choices, = np.where(self.possible_choices != self.s)
        choices = self.np_random.choice(possible_choices, size=n_choices, replace=False)
        *self.random_states, _ = np.split(
            choices, np.cumsum(list(self.random.values()))
        )

        self.assign(**dict(zip(self.random.keys(), self.random_states)))

    def reset(self):
        o = super().reset()
        self.set_randoms()
        self.last_transition = None
        return self.append_randoms(o)

    def step(self, a):
        s, r, t, i = super().step(a)
        return self.append_randoms(s), r, t, i


if __name__ == "__main__":
    from gridworld_env.random_walk import run
    import gym

    run(gym.make("1x3RandomGridWorld-v0"))
    # env.reset()
    # while True:
    #     s, r, t, i = env.step(env.action_space.sample())
    #     env.render()
    #     print('reward', r)
    #     time.sleep(.5)
    #     if t:
    #         print('reset')
    #         time.sleep(1)
    #         env.reset()
