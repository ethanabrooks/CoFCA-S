from abc import ABC
from collections import defaultdict
from typing import Collection

import numpy as np


class Aggregator(ABC):
    def update(self, *args, **kwargs):
        raise NotImplementedError

    def items(self):
        raise NotImplementedError


class EpisodeAggregator(Aggregator):
    def __init__(self):
        self.complete_episodes = defaultdict(list)
        self.incomplete_episodes = defaultdict(list)

    def update(self, dones: Collection[bool], **values):
        values.update(time_steps=[1 for _ in dones])
        for k, vs in values.items():
            incomplete_episodes = self.incomplete_episodes[k]
            if not incomplete_episodes:
                incomplete_episodes = self.incomplete_episodes[k] = [[] for _ in vs]
            assert len(incomplete_episodes) == len(vs) == len(dones)
            for i, (value, done) in enumerate(zip(vs, dones)):
                incomplete_episodes[i].append(value)
                if done:
                    self.complete_episodes[k].append(sum(incomplete_episodes[i]))
                    incomplete_episodes[i] = []

    def items(self):
        for k, v in self.complete_episodes.items():
            yield k, np.mean(v)


class InfosAggregator(EpisodeAggregator):
    def update(self, *infos: dict, dones: Collection[bool]):
        assert len(dones) == len(infos)
        for i, (done, info) in enumerate(zip(dones, infos)):
            for k, v in info.items():
                incomplete_episodes = self.incomplete_episodes[k]
                if not incomplete_episodes:
                    incomplete_episodes = self.incomplete_episodes[k] = [
                        [] for _ in infos
                    ]
                incomplete_episodes[i].append(v)
                if done:
                    self.complete_episodes[k].append(sum(incomplete_episodes[i]))
                    incomplete_episodes[i] = []


class EvalWrapper(Aggregator):
    def __init__(self, aggregator: Aggregator):
        self.aggregator = aggregator

    def update(self, *args, **kwargs):
        self.aggregator.update(*args, **kwargs)

    def items(self):
        for k, v in self.aggregator.items():
            yield "eval_" + k, v
