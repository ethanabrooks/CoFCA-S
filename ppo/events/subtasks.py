import abc
import re
import numpy as np
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
    Cat,
)


class Subtask:
    def step(self, *interactions, **objects):
        pass

    @abc.abstractmethod
    def condition(self, *interactions, **objects):
        raise NotImplementedError

    @property
    def reward(self):
        return -0.1

    def __str__(self):
        return " ".join(
            re.findall("[A-Z][^A-Z]*", self.__class__.__name__)
        ).capitalize()


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

    @property
    def reward(self):
        return 1


class CatchMouse(Subtask):
    def condition(self, *interactions, mouse: Mouse, **objects):
        return mouse in interactions

    @property
    def reward(self):
        return 1


class ComfortBaby(Subtask):
    def __init__(self):
        self.time_crying = 0
        super().__init__()

    def step(self, *interactions, baby: Baby, **objects):
        if baby.activated:
            self.time_crying += 1
        if baby in interactions:
            self.time_crying = 0

    def condition(self, *interactions, baby: Baby, **objects):
        return baby.activated


class MakeDinner(Subtask):
    def __init__(self):
        self.made = 0
        super().__init__()

    def condition(self, *interactions, food: Food, table: Table, **objects):
        return food.pos == table.pos and food.activated

    @property
    def reward(self):
        return 1


class MakeFire(Subtask):
    def condition(self, *interactions, fire: Fire, **objects):
        return fire.activated

    @property
    def reward(self):
        return 0.01


class KillFlies(Subtask):
    def condition(self, *interactions, fly: List[Fly], **objects):
        print([f.pos for f in fly if f.activated])
        return any(f.activated for f in fly)


class CleanMess(Subtask):
    def condition(self, *interactions, mess: List[Mess], **objects):
        return any(m.activated for m in mess)


class AvoidDog(Subtask):
    def __init__(self, min_range):
        self.range = min_range

    def condition(self, *interactions, dog: Dog, agent: Agent, **objects):
        return (
            dog.pos is not None
            and np.linalg.norm(np.array(dog.pos) - np.array(agent.pos)) < self.range
        )


class LetDogIn(Subtask):
    def __init__(self, max_time_outside):
        self.max_time_outside = max_time_outside
        self.time_outside = None

    def step(self, *interactions, dog: Dog, **objects):
        if dog.pos is None:
            self.time_outside += 1
        else:
            self.time_outside = 0

    def condition(self, *interactions, dog: Dog, agent: Agent, **objects):
        return self.time_outside > self.max_time_outside


class WatchBaby(Subtask):
    def __init__(self, max_range: int):
        self.range = max_range

    def condition(self, *interactions, baby: Baby, agent: Agent, **objects):
        return np.linalg.norm(np.array(baby.pos) - np.array(agent.pos)) > self.range


class KeepBabyOutOfFire(Subtask):
    def condition(self, *interactions, baby: Baby, fire: Fire, **objects):
        return fire.activated and baby.pos == fire.pos

    @property
    def reward(self):
        return -2


class KeepCatFromDog(Subtask):
    def condition(self, *interactions, cat: Cat, dog: Dog, **objects):
        return cat.pos == dog.pos

    @property
    def reward(self):
        return -1
