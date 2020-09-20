from collections import defaultdict

import numpy as np


class EpochCounter:
    last_return_name = "last_return"

    def __init__(self, num_processes):
        self.episode_rewards = []
        self.episode_time_steps = []
        self.rewards = np.zeros(num_processes)
        self.time_steps = np.zeros(num_processes)
        self.infos_name = defaultdict(list)
        self.last_return = 0

    def update(self, reward, done, infos):
        self.rewards += reward.numpy()
        self.time_steps += np.ones_like(done)
        self.episode_rewards += list(self.rewards[done])
        self.episode_time_steps += list(self.time_steps[done])
        self.rewards[done] = 0
        self.time_steps[done] = 0
        for info in infos:
            for k, v in info.items():
                self.infos_name[k].append(v)

    def reset(self):
        self.last_return = np.mean(self.episode_rewards)
        self.episode_rewards = []
        self.episode_time_steps = []
        self.infos_name = defaultdict(list)

    def items(self, prefix=""):
        if self.episode_rewards:
            yield prefix + "epoch_returns", np.mean(self.episode_rewards)
        if self.episode_time_steps:
            yield prefix + "epoch_time_steps", np.mean(self.episode_time_steps)
        yield prefix + self.last_return_name, self.last_return
        for k, vs in self.infos_name.items():
            yield prefix + k, np.mean(vs)
