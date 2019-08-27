import functools
import itertools
import re
from abc import ABC
from copy import copy
from typing import List

import numpy as np


def add_pair(a, b):
    a1, a2 = a
    b1, b2 = b
    return a1 + b1, a2 + b2


def subtract_pair(a, b):
    a1, a2 = a
    b1, b2 = b
    return a1 - b1, a2 - b2


def distance(a, b):
    p1, p2 = subtract_pair(a, b)
    return p1 ** 2 + p2 ** 2


class Object:
    actions = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]

    def __init__(self, objects: List, random: np.random, height: int, width: int):
        self.random = random
        self.width = width
        self.height = height
        self.objects = objects  # type: List[Object]
        self.pos = None
        self.activated = False
        self.grasped = False
        self.excluded_positions = [
            o.pos for o in self.objects if type(o) is type(self) and o is not self
        ]
        self.random_thresholds = []

    def reset(self):
        self.activated = False
        self.grasped = False

    def step(self, **kwargs):
        if self.pos is None:
            return
        p1, p2 = add_pair(self.pos, self.action(**kwargs))
        if (p1, p2) not in self.excluded_positions:
            self.pos = min(max(p1, 0), self.height - 1), min(max(p2, 0), self.width - 1)

    def interact(self):
        pass

    def get_objects(self, *types):
        return (o for o in self.objects if type(o) in types)

    def get_object(self, *types):
        objects = (o for o in self.objects if type(o) in types)
        try:
            return next(objects)
        except StopIteration:
            raise RuntimeError("Object of type", types, "not found.")

    def icon(self):
        return str(self)[:1]

    def action(self, agent_action, **kwargs):
        return agent_action

    def random_action(self, actions):
        return actions[self.random.choice(len(actions))]

    def move_toward(self, toward, actions):
        try:
            pos = toward.pos
        except AttributeError:
            pos = self.get_object(toward).pos
        return min(actions, key=lambda a: distance(add_pair(self.pos, a), pos))

    def __str__(self):
        return "_".join(re.findall("[A-Z][^A-Z]*", self.__class__.__name__)).lower()


class Immobile(Object, ABC):
    def action(self, **kwargs):
        return 0, 0


class RandomPosition(Object, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.candidate_positions = list(self.get_candidate_positions())
        self.pos = None

    def reset(self):
        self.pos = self.random_position()
        super().reset()

    def get_candidate_positions(self):
        return itertools.product(range(self.height), range(self.width))

    def random_position(self):
        other = list(self.excluded_positions)
        available = [t for t in self.candidate_positions if t not in other]
        if not available:
            return None
        choice = self.random.choice(len(available))
        return tuple(available[choice])


class Slow(Object, ABC):
    def __init__(self, speed, **kwargs):
        super().__init__(**kwargs)
        self.random_thresholds += [speed]

    def action(self, coin_flips, **kwargs):
        move, *coin_flips = coin_flips
        if not move:
            return 0, 0
        return super().action(coin_flips=coin_flips, **kwargs)


class Graspable(Object, ABC):
    def interact(self):
        self.grasped = not self.grasped
        super().interact()

    def action(self, agent_action, **kwargs):
        if self.grasped:
            return agent_action
        else:
            return super().action(agent_action=agent_action, **kwargs)


class Activating(Object, ABC):
    def activate(self):
        self.activated = True

    def deactivate(self):
        self.activated = False


class Deactivatable(Activating, ABC):
    def interact(self):
        if self.activated:
            self.deactivate()
        super().interact()


class Activatable(Activating, ABC):
    def interact(self):
        if not self.activated:
            self.activate()
        super().interact()


class RandomActivating(Activating, ABC):
    def __init__(self, activation_prob, **kwargs):
        super().__init__(**kwargs)
        self.activation_prob = activation_prob

    def step(self, coin_flips, **kwargs):
        activate, *coin_flips = coin_flips
        if not self.activated and activate:
            self.activate()
        return super().step(coin_flips=coin_flips, **kwargs)


class Wall(Immobile, RandomPosition, ABC):
    def get_candidate_positions(self):
        for i in range(self.height):
            yield i, 0
            yield i, self.width - 1
        for j in range(self.width):
            yield 0, j
            yield self.height - 1, j


class Agent(RandomPosition):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grasping = None

    def reset(self):
        self.grasping = None
        super().reset()

    def icon(self):
        return "😀"


class Door(Wall, Deactivatable, Immobile):
    def __init__(self, time_limit, **kwargs):
        super().__init__(**kwargs)
        self.time_to_ring = self.random.choice(time_limit // 2)
        self.t = 0

    def reset(self):
        self.t = 0
        super().reset()

    def step(self, **kwargs):
        self.t += 1
        self.activated = False
        if self.t == self.time_to_ring:
            self.activated = True
        return super().step(**kwargs)

    def interact(self):
        dog = self.get_object(Dog)
        if dog.pos is None:
            dog.pos = self.pos

    def icon(self):
        return "🔔" if self.activated else "🚪"


class MouseHole(Wall):
    def icon(self):
        return "🕳"


class Mouse(Slow):
    def __init__(self, speed, toward_hole_prob, mouse_prob, **kwargs):
        super().__init__(speed=speed, **kwargs)
        self.random_thresholds += [mouse_prob, speed, toward_hole_prob]
        self.caught = False

    def reset(self):
        self.caught = False
        super().reset()

    def interact(self):
        self.pos = None
        self.caught = True
        super().interact()

    def step(self, coin_flips, **kwargs):
        hole = self.get_object(MouseHole)
        enter_hole, appear, *coin_flips = coin_flips
        if enter_hole and self.pos == hole.pos:
            self.pos = None
        if appear and self.pos is None:
            self.pos = hole.pos
        return super().step(coin_flips=coin_flips, **kwargs)

    def action(self, actions, coin_flips, **kwargs):
        toward_hole, *coin_flips = coin_flips
        if self.pos is None:
            return
        if toward_hole:
            # step toward hole
            return self.move_toward(MouseHole, actions)
        else:
            return self.random_action(actions)

    def icon(self):
        return "🐁"


class Baby(Graspable, Slow, RandomPosition, RandomActivating, Deactivatable):
    def __init__(self, toward_fire_prob, **kwargs):
        super().__init__(**kwargs)
        self.random_thresholds += [toward_fire_prob]

    def interact(self):
        if self.activated:
            self.deactivate()
        super().interact()

    def action(self, actions, coin_flips, **kwargs):
        fire = self.get_object(Fire)
        toward_fire, *coin_flips = coin_flips
        if fire.activated and toward_fire:
            return self.move_toward(fire, actions)
        else:
            return self.random_action(actions)

    def icon(self):
        return "😭" if self.activated else "👶"


class Oven(RandomPosition, Activatable, Immobile):
    def __init__(self, time_to_heat, **kwargs):
        super().__init__(**kwargs)
        self.time_to_heat = time_to_heat
        self.time_heating = 0

    def reset(self):
        self.time_heating = 0
        super().reset()

    def hot(self):
        return self.time_heating > self.time_to_heat

    def interact(self):
        if self.activated:
            self.deactivate()
        else:
            self.activate()
        super().interact()

    def step(self, **kwargs):
        if self.activated:
            self.time_heating += 1
        super().step(**kwargs)

    def deactivate(self):
        self.time_heating = 0
        super().deactivate()

    def icon(self):
        return "♨️" if self.hot() else "𝌱"


class Refrigerator(RandomPosition, Immobile):
    def icon(self):
        return "❄️"


class Table(RandomPosition, Immobile):
    def icon(self):
        return "🍽"


class Food(Graspable, RandomPosition, Immobile, Activating):
    def __init__(self, cook_time, **kwargs):
        super().__init__(**kwargs)
        self.cook_time = cook_time
        self.time_cooking = 0

    def reset(self):
        self.time_cooking = 0
        super().reset()

    def icon(self):
        return "🍳" if self.activated else "🥚"

    def step(self, **kwargs):
        # noinspection PyTypeChecker
        oven = self.get_object(Oven)  # type: Oven
        if self.time_cooking == self.cook_time:
            self.activate()
        if self.pos == oven.pos and oven.hot():
            self.time_cooking += 1
        super().step(**kwargs)


class Dog(Graspable, Slow, RandomPosition):
    def __init__(self, toward_cat_prob, **kwargs):
        super().__init__(**kwargs)
        self.let_out = False
        self.random_thresholds += [toward_cat_prob]

    def reset(self):
        self.let_out = False
        super().reset()

    def interact(self):
        door = self.get_object(Door)
        if self.pos == door.pos and not self.let_out:
            self.pos = None
            self.let_out = True
        super().interact()

    def action(self, coin_flips, actions, **kwargs):
        toward_cat, *coin_flips = coin_flips
        return self.move_toward(Cat if toward_cat else Agent, actions)

    def icon(self):
        return "🐕"


class Cat(Graspable, Slow, RandomPosition):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.let_out = False

    def action(self, actions, **kwargs):
        if self.grasped:
            return super().action(**kwargs)
        else:
            actions = [a for a in self.actions if a != self.move_toward(Agent, actions)]
            dog = self.get_object(Dog)
            if dog.pos is None:
                return self.random_action(actions)
            return self.move_toward(Dog, actions)

    def icon(self):
        return "🐈"


class Mess(Immobile):
    def __init__(self, pos, **kwargs):
        super().__init__(**kwargs)
        self.eventual_pos = pos

    def step(self, coin_flips, **kwargs):
        exist, *coin_flips = coin_flips
        dog = self.get_object(Dog)
        if dog.pos == self.eventual_pos:
            self.pos = dog.pos
        super().step(coin_flips=coin_flips, **kwargs)

    def interact(self):
        self.pos = None
        super().interact()

    def icon(self):
        return "💩"


class Fire(RandomPosition, Immobile, Activatable):
    def icon(self):
        return "🔥" if self.activated else "🜂"


class Fly(RandomPosition, Deactivatable):
    def __init__(self, pos, **kwargs):
        super().__init__(**kwargs)
        self.n_children = 0
        objects = copy(self.objects)
        self.random.shuffle(objects)
        self.parent = next(
            (o for o in objects if type(o) is Fly and o.n_children < 2), None
        )
        if self.parent:
            self.parent.n_children += 1
        self.eventual_pos = pos
        assert (self.pos is not None) == self.activated

    def reset(self):
        if self.parent is None:
            self.pos = self.random_position()
        else:
            self.pos = None
        super().reset()

    def step(self, coin_flips, **kwargs):
        activate, *coin_flips = coin_flips
        if activate and self.parent is not None:
            self.pos = self.random_position()
        super().step(coin_flips=coin_flips, **kwargs)

    def interact(self):
        self.pos = None
        super().interact()

    def icon(self):
        return "🦟"
