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
    Oven,
)


class Instruction:
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


class AnswerDoor(Instruction):
    def __init__(self, time_limit=10):
        self.time_limit = time_limit
        self.time_waiting = None

    def step(self, *interactions, door: Door, **objects):
        if door.activated:
            self.time_waiting = 0
        if self.time_waiting is not None:
            self.time_waiting += 1
            if door in interactions and self.time_waiting < self.time_limit:
                self.time_waiting = None

    def condition(self, *interactions, door: Door, **objects):
        if self.time_waiting is None:
            return False
        return self.time_waiting > self.time_limit


class CatchMouse(Instruction):
    def condition(self, *interactions, mouse: Mouse, **objects):
        return mouse.pos is not None


class ComfortBaby(Instruction):
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


class MakeDinner(Instruction):
    def __init__(self):
        self.made = 0
        super().__init__()

    def condition(self, *interactions, food: Food, table: Table, **objects):
        # return not (food.pos == table.pos and food.activated)
        return not food.activated


class MakeFire(Instruction):
    def condition(self, *interactions, fire: Fire, baby: Baby, **objects):
        return not fire.activated or (fire.activated and baby.pos == fire.pos)


class KillFlies(Instruction):
    def condition(self, *interactions, fly: List[Fly], **objects):
        return any(f.activated for f in fly)


class CleanMess(Instruction):
    def condition(self, *interactions, mess: List[Mess], **objects):
        return any(m.activated for m in mess)


class AvoidDog(Instruction):
    def __init__(self, min_range):
        self.range = min_range

    def condition(self, *interactions, dog: Dog, agent: Agent, **objects):
        return (
            dog.pos is not None
            and np.linalg.norm(np.array(dog.pos) - np.array(agent.pos)) < self.range
        )


class LetDogIn(Instruction):
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


class WatchBaby(Instruction):
    def __init__(self, max_range: int):
        self.range = max_range

    def condition(self, *interactions, baby: Baby, agent: Agent, **objects):
        return np.linalg.norm(np.array(baby.pos) - np.array(agent.pos)) > self.range


class KeepBabyOutOfFire(Instruction):
    def condition(self, *interactions, baby: Baby, fire: Fire, **objects):
        return fire.activated and baby.pos == fire.pos


class KeepCatFromDog(Instruction):
    def condition(self, *interactions, cat: Cat, dog: Dog, **objects):
        return cat.pos == dog.pos
