import functools
import re
from abc import ABC
from typing import List

import numpy as np


class Object:
    def __init__(
        self,
        objects: List,
        random: np.random,
        height: int,
        width: int,
        object_types: List[type],
    ):
        self.random = random
        self.width = width
        self.height = height
        self.objects = objects  # type: List[Object]
        self.pos = None
        self.activated = False
        self.grasped = False
        self.obstacle_types = [type(self)]

    @property
    def obstacle(self):
        return False

    def step(self, action):
        if self.pos is None:
            return

        action = self.wrap_action(action)
        pos = np.array(self.pos) + np.array(action)
        if tuple(pos) in (
            o.pos for o in self.objects if type(o) in self.obstacle_types
        ):
            pos = self.pos
        self.pos = tuple(
            np.clip(
                pos, np.zeros(2, dtype=int), np.array([self.height, self.width]) - 1
            )
        )

    def interact(self):
        pass

    def other_positions(self):
        for o in self.objects:
            if o is not self:
                yield o.pos

    def get_objects(self, *types):
        return (o for o in self.objects if type(o) in types)

    def get_object(self, *types):
        objects = (o for o in self.objects if type(o) in types)
        try:
            return next(objects)
        except StopIteration:
            raise RuntimeError("Object of type", types, "not found.")

    def __str__(self):
        return "_".join(re.findall("[A-Z][^A-Z]*", self.__class__.__name__)).lower()

    def icon(self):
        return str(self)[:1]

    def wrap_action(self, action):
        return action


class Immobile(Object, ABC):
    def wrap_action(self, action):
        return np.zeros(2, dtype=int)


class RandomPosition(Object, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pos = self.random_position()

    def random_position(self):
        available = [
            t
            for t in map(tuple, self.candidate_positions())
            if t not in self.other_positions()
        ]
        if not available:
            return None
        choice = self.random.choice(len(available))
        return tuple(available[choice])

    def candidate_positions(self):
        return (
            np.stack(np.meshgrid(np.arange(self.width), np.arange(self.height)))
            .reshape(2, -1)
            .T
        )


class RandomWalking(Object, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)]

    def wrap_action(self, action):
        choice = self.random.choice(len(self.actions))
        return self.actions[choice] if self.random.rand() < 0.7 else (0, 0)


class Graspable(Object, ABC):
    def interact(self):
        self.grasped = not self.grasped
        super().interact()

    def wrap_action(self, action):
        if self.grasped:
            return action
        else:
            return super().wrap_action(action)


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


class DurationActivating(Activating, ABC):
    def __init__(self, duration, **kwargs):
        super().__init__(**kwargs)
        self.duration = duration
        self.time_activated = 0

    def step(self, action):
        if self.time_activated == self.duration:
            self.deactivate()
        if self.activated:
            self.time_activated += 1
        return super().step(action)

    def deactivate(self):
        self.time_activated = 0
        super().deactivate()


class RandomActivating(Activating, ABC):
    def __init__(self, activation_prob, **kwargs):
        super().__init__(**kwargs)
        self.activation_prob = activation_prob

    def step(self, action):
        if not self.activated:
            rand = self.random.rand()
            if rand < self.activation_prob:
                self.activate()
        return super().step(action)


class Wall(Immobile, RandomPosition, ABC):
    def candidate_positions(self):
        idxs = np.stack(np.meshgrid(np.arange(self.width), np.arange(self.height))).T
        return np.concatenate(
            [
                idxs[:, 0],  # right
                idxs[:, -1],  # left
                idxs[0],  # top
                idxs[-1],  # bottom
            ]
        )


class Agent(RandomPosition):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grasping = None

    def icon(self):
        return "😀"


class Door(Wall, RandomActivating, Deactivatable, Immobile):
    def step(self, action):
        self.activated = False
        return super().step(action)

    def interact(self):
        dog = self.get_object(Dog)
        if dog.pos is None:
            dog.pos = self.pos

    def icon(self):
        return "🔔" if self.activated else "🚪"


class MouseHole(Wall):
    def icon(self):
        return "🕳"


class Mouse(RandomActivating, RandomWalking, Deactivatable):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.caught = False

    def activate(self):
        if not self.caught:
            self.pos = self.get_object(MouseHole).pos
            super().activate()

    def deactivate(self):
        self.pos = None
        self.caught = True
        super().deactivate()

    @property
    def hole(self):
        return self.get_object(MouseHole)

    def wrap_action(self, action):
        if self.pos not in (None, self.hole.pos) and self.random.rand() < 0.5:
            # step toward hole
            from_hole = np.array(self.pos) - np.array(self.hole.pos)
            action = min(self.actions, key=lambda a: np.sum(np.abs(a + from_hole)))
        return action

    def step(self, action):
        hole = self.get_object(MouseHole)
        if self.pos == hole.pos:
            self.pos = None
            self.activated = False
        return super().step(action)

    def icon(self):
        return "🐁"


class Baby(Graspable, RandomPosition, RandomActivating, RandomWalking, Deactivatable):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def interact(self):
        if self.activated:
            self.deactivate()
        super().interact()

    def icon(self):
        return "😭" if self.activated else "👶"


class Oven(RandomPosition, Activatable, Immobile):
    def __init__(self, time_to_heat, **kwargs):
        super().__init__(**kwargs)
        self.time_to_heat = time_to_heat
        self.time_heating = 0

    def hot(self):
        return self.time_heating > self.time_to_heat

    def interact(self):
        if self.activated:
            self.deactivate()
        else:
            self.activate()
        super().interact()

    def step(self, action):
        if self.activated:
            self.time_heating += 1
        super().step(action)

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


class Food(Graspable, Immobile, Activating):
    def __init__(self, cook_time, **kwargs):
        super().__init__(**kwargs)
        # noinspection PyTypeChecker
        # noinspection PyTypeChecker
        refrigerator = self.get_object(Refrigerator)  # type: Refrigerator
        self.cook_time = cook_time
        self.time_cooking = 0
        self.pos = refrigerator.pos

    def icon(self):
        return "🍳" if self.activated else "🥚"

    def step(self, action):
        # noinspection PyTypeChecker
        oven = self.get_object(Oven)  # type: Oven
        if self.time_cooking == self.cook_time:
            self.activate()
        if self.pos == oven.pos and oven.hot():
            self.time_cooking += 1
        super().step(action)


class Dog(Graspable, RandomPosition, RandomWalking):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.let_out = False

    def interact(self):
        door = self.get_object(Door)
        if self.pos == door.pos and not self.let_out:
            self.pos = None
            self.let_out = True
        super().interact()

    def icon(self):
        return "🐕"


class Cat(Graspable, RandomPosition, RandomWalking):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.let_out = False

    def wrap_action(self, action):
        if self.grasped:
            return super().wrap_action(action)
        else:
            agent = self.get_object(Agent)
            from_agent = np.array(self.pos) - np.array(agent.pos)
            toward_agent = min(
                self.actions, key=lambda a: np.sum(np.abs(np.array(a) + from_agent))
            )
            actions = list(set(self.actions) - {toward_agent})
            choice = self.random.choice(len(actions))
            return super().wrap_action(actions[choice])

    def icon(self):
        return "🐈"


class Mess(Immobile, RandomActivating, Deactivatable):
    def activate(self):
        messes = self.get_objects(Mess)
        dog = self.get_object(Dog)
        if dog.pos not in {m.pos for m in messes}:
            self.pos = dog.pos
            super().activate()

    def deactivate(self):
        self.pos = None
        super().deactivate()

    def icon(self):
        return "💩"


class Fire(RandomPosition, Immobile, Activatable):
    def icon(self):
        return "🔥" if self.activated else "🜂"


class Fly(RandomPosition, RandomWalking, RandomActivating, Deactivatable):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            self.parent = self.get_object(Fly)
            self.activated = False
            self.pos = None
        except RuntimeError:
            self.parent = None
            self.activated = True
        assert (self.pos is not None) == self.activated

    def deactivate(self):
        self.pos = None
        super().deactivate()

    def activate(self):
        if self.parent is not None:
            self.pos = self.random_position()
            super().activate()

    def icon(self):
        return "🦟"
