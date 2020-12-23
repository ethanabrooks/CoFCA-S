import time
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Collection, Generator, Tuple, Dict, Optional

import numpy as np


class Aggregator(ABC):
    def update(self, *args, **kwargs):
        raise NotImplementedError

    def items(self) -> Generator[Tuple[str, any], None, None]:
        raise NotImplementedError


class Timer:
    def __init__(self):
        self.count = 0
        self.total = 0
        self.last_tick = time.time()

    def update(self):
        self.count += 1
        tick = time.time()
        self.total = self.total + tick - self.last_tick
        self.last_tick = tick

    def average(self):
        if self.total == self.count == 0:
            return None
        return self.total / self.count

    def tick(self):
        self.last_tick = time.time()


class TimeKeeper:
    def __init__(self):
        self.timers = defaultdict(Timer)
        self.yield_average = {}

    def __getitem__(self, item):
        return self.timers[item]

    @abstractmethod
    def items(self) -> Generator[Tuple[str, any], None, None]:
        pass


class TotalTimeKeeper(TimeKeeper):
    def items(self) -> Generator[Tuple[str, any], None, None]:
        for k, v in self.timers.items():
            yield f"time spent {k}", v.total


class AverageTimeKeeper(TimeKeeper):
    def items(self) -> Generator[Tuple[str, any], None, None]:
        for k, v in self.timers.items():
            average = v.average()
            if average is not None:
                yield f"time per {k}", average


class EpisodeAggregator(Aggregator):
    def __init__(self):
        self.complete_episodes = defaultdict(list)
        self.incomplete_episodes = defaultdict(list)

    def update(self, dones: Collection[bool], **values):
        values.update({"time steps": [1 for _ in dones]})
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
            self.log_info(i, done, info, len(infos))

    def log_info(self, i, done, info, n):
        for k, v in info.items():
            incomplete_episodes = self.incomplete_episodes[k]
            if not incomplete_episodes:
                incomplete_episodes = self.incomplete_episodes[k] = [
                    [] for _ in range(n)
                ]
            incomplete_episodes[i].append(v)
            if done:
                self.complete_episodes[k].append(sum(incomplete_episodes[i]))
                incomplete_episodes[i] = []


class EvalAggregator(Aggregator, ABC):
    def __init__(self):
        super().__init__()
        self.complete = set()

    def items(self) -> Generator[Tuple[str, any], None, None]:
        for k, v in super().items():
            yield "eval " + k, v


class EvalEpisodeAggregator(EvalAggregator, EpisodeAggregator):
    def update(self, dones: Collection[bool], **values):
        values.update({"time steps": [1 for _ in dones]})
        for k, vs in values.items():
            incomplete_episodes = self.incomplete_episodes[k]
            if not incomplete_episodes:
                incomplete_episodes = self.incomplete_episodes[k] = [[] for _ in vs]
            assert len(incomplete_episodes) == len(vs) == len(dones)
            for i, (value, done) in enumerate(zip(vs, dones)):
                if i in self.complete:
                    continue
                if done:
                    self.complete.add(i)
                incomplete_episodes[i].append(value)
                if done:
                    self.complete_episodes[k].append(sum(incomplete_episodes[i]))
                    incomplete_episodes[i] = []


class EvalInfosAggregator(EvalAggregator, InfosAggregator):
    def update(self, *infos: dict, dones: Collection[bool]):
        assert len(dones) == len(infos)
        for i, (done, info) in enumerate(zip(dones, infos)):
            if i in self.complete:
                continue
            if done:
                self.complete.add(i)
            self.log_info(i, done, info, len(infos))
