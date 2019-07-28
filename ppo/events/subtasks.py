import abc
from typing import List

from ppo.events.objects import (
    Door,
    Mouse,
    Baby,
    Table,
    Food,
    Fire,
    Fly,
    Mess,
    Dog,
    Agent,
)


class Subtask:
    def step(self, *interactions, **objects):
        pass

    @abc.abstractmethod
    def condition(self, *interactions, **objects):
        raise NotImplementedError

    @property
    def reward_delta(self):
        return 1


class AnswerDoor(Subtask):
    def __init__(self, time_limit=10):
        self.time_since_ring = 0
        self.time_limit = time_limit

    def step(self, *interactions, door: Door, **objects):
        self.time_since_ring += 1
        if door.activated:  # door bell rings once
            self.time_since_ring = 0

    def condition(self, *interactions, door: Door, **objects):
        return self.time_since_ring < self.time_limit and door in interactions


class CatchMouse(Subtask):
    def condition(self, *interactions, mouse: Mouse, **objects):
        return mouse in interactions


class ComfortBaby(Subtask):
    def __init__(self):
        self.time_crying = 0

    def step(self, *interactions, baby: Baby, **objects):
        if baby.activated:
            self.time_crying += 1
        if baby in interactions:
            self.time_crying = 0

    def condition(self, *interactions, baby: Baby, **objects):
        return baby in interactions


class MakeDinner(Subtask):
    def condition(self, reward, *interactions, food: Food, table: Table, **objects):
        return {food, table}.issubset(interactions) and food.cooked


class MakeFire(Subtask):
    def condition(self, reward, *interactions, fire: Fire, **objects):
        return fire.activated

    @property
    def reward_delta(self):
        return 0.1


class KillFlies(Subtask):
    def condition(self, reward, *interactions, flies: List[Fly], **objects):
        return any(fly in interactions for fly in flies)


class CleanMess(Subtask):
    def condition(self, reward, *interactions, messes: List[Mess], **objects):
        return any(mess in interactions for mess in messes)


class AvoidDog(Subtask):
    def __init__(self, min_range):
        self.range = min_range

    def condition(self, reward, *interactions, dog: Dog, agent: Agent, **objects):
        return dog.pos - agent.pos < self.range

    @property
    def reward_delta(self):
        return -0.1


class WatchBaby(Subtask):
    def __init__(self, max_range: int):
        self.range = max_range

    def condition(self, reward, *interactions, baby: Baby, agent: Agent, **objects):
        return baby.pos - agent.pos > self.range

    @property
    def reward_delta(self):
        return -0.1
