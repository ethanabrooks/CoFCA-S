import time
from abc import ABC
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Collection, Generator, Tuple

import numpy as np


class Aggregator(ABC):
    def update(self, *args, **kwargs):
        raise NotImplementedError

    def items(self) -> Generator[Tuple[str, any], None, None]:
        raise NotImplementedError


class TimeAggregator(Aggregator):
    def __init__(self):
        self.count = 0
        self.time = 0
        self.tick = time.time()

    def update(self):
        self.count += 1
        self.time = self.time + time.time() - self.tick

    def items(self):
        yield "time", self.average()

    def average(self):
        if self.time == self.count == 0:
            return 0
        return self.time / self.count

    def reset(self):
        self.tick = time.time()


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

    def items(self) -> Generator[Tuple[str, any], None, None]:
        for k, v in self.complete_episodes.items():
            yield k, np.mean(v)

    def reset(self):
        self.complete_episodes = defaultdict(list)


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


@dataclass(frozen=True)
class EvalWrapper(Aggregator):
    aggregator: Aggregator

    def update(self, *args, **kwargs):
        self.aggregator.update(*args, **kwargs)

    def items(self) -> Generator[Tuple[str, any], None, None]:
        for k, v in self.aggregator.items():
            yield "eval_" + k, v
