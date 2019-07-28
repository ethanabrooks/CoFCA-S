from abc import ABC
from typing import List

import numpy as np


class Object:
    def __init__(self, random: np.random, objects: List, height: int, width: int):
        self.width = width
        self.height = height
        self.objects = objects  # type: List[Object]
        self.np_random = random
        self.pos = None

    @property
    def obstacle(self):
        return False

    def step(self, action):
        if self.pos is None:
            return
        pos = self.pos + np.array(action)
        if pos in [o.pos for o in self.get_objects(self.__class__)]:
            pos = self.pos
        self.pos = np.clip(pos, np.zeros(2), np.array([self.height, self.width]))

    def reset(self):
        pass

    def interact(self):
        pass

    def other_positions(self):
        return [tuple(o.pos) for o in self.objects if o is not self]

    def get_objects(self, types):
        return (o for o in self.objects if isinstance(o, types))

    def get_object(self, types):
        objects = (o for o in self.objects if isinstance(o, types))
        try:
            return next(objects)
        except StopIteration:
            raise RuntimeError("Object of type", types, "not found.")


class Immobile(Object, ABC):
    def step(self, action):
        return super().step(np.zeros(2))


class RandomPosition(Object, ABC):
    def reset(self):
        available = [
            t
            for t in map(tuple, self.candidate_positions())
            if t not in self.other_positions()
        ]
        choice = self.np_random.choice(len(available))
        self.pos = available[choice]
        return super().reset()

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

    def step(self, action):
        choice = self.np_random.choice(len(self.actions))
        return super().step(self.actions[choice])


class Graspable(Object, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grasped = False

    def interact(self):
        if self.pos not in self.other_positions():
            self.grasped = not self.grasped
        super().interact()

    def step(self, action):
        return super().step(action if self.grasped else np.zeros(2))


class Activating(Object, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.activated = False

    def activate(self):
        self.activated = True

    def deactivate(self):
        self.activated = False


class Deactivatable(Activating, ABC):
    def interact(self):
        if self.activated:
            self.deactivate()
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
            if self.np_random.random() < self.activation_prob:
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
    pass


class Door(Wall, RandomActivating, Deactivatable, Immobile):
    def step(self, action):
        self.activated = False
        return super().step(action)


class MouseHole(Wall):
    pass


class Mouse(RandomActivating, RandomWalking, Deactivatable):
    def activate(self):
        self.pos = self.get_object(MouseHole)
        super().activate()

    def deactivate(self):
        self.pos = None

    def step(self, action):
        if self.np_random.random() < 0.2:
            # step toward hole
            hole = self.get_object(MouseHole)
            action = min(
                self.actions, key=lambda a: np.sum(np.abs(self.pos + a - hole))
            )
        return super().step(action)


class Baby(RandomActivating, RandomWalking, Deactivatable, Graspable):
    def interact(self):
        if self.activated:
            self.deactivate()
        else:
            Graspable.interact(self)
        super().interact()


class Oven(Object, Activating, Immobile):
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


class Refrigerator(Immobile):
    pass


class Table(Immobile):
    pass


class Food(Graspable):
    def __init__(self, cook_time, **kwargs):
        super().__init__(**kwargs)
        # noinspection PyTypeChecker
        self.oven = self.get_object(Oven)  # type: Oven
        # noinspection PyTypeChecker
        self.refrigerator = self.get_object(Refrigerator)  # type: Refrigerator
        self.cooked = False
        self.cook_time = cook_time
        self.time_cooking = 0

    def reset(self):
        self.pos = self.refrigerator.pos
        super().reset()

    def step(self, action):
        if self.time_cooking == self.cook_time:
            self.cooked = True
        if self.pos == self.oven.pos and self.oven.hot():
            self.time_cooking += 1


class Dog(RandomWalking, Graspable):
    def interact(self):
        door = self.get_object(Door)
        if self.grasped and self.pos == door.pos:
            self.pos = None


class Cat(RandomWalking, Graspable):
    def step(self, action):
        agent = self.get_object(Agent)
        toward_agent = min(
            self.actions, key=lambda a: np.sum(np.abs(self.pos + a - agent))
        )
        actions = list(set(self.actions) - {toward_agent})
        choice = self.np_random.choice(len(actions))
        return super().step(actions[choice])


class Mess(Immobile, RandomActivating, Deactivatable):
    def activate(self):
        messes = self.get_objects(Mess)
        dog = self.get_object(Dog)
        if dog.pos not in {m.pos for m in messes}:
            self.pos = dog.pos

    def deactivate(self):
        self.pos = None


class Fire(Immobile, Activating):
    pass


class Fly(RandomWalking, RandomActivating, Deactivatable):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            self.parent = self.get_object(Fly)
        except RuntimeError:
            self.parent = None

    def step(self, action):
        if not self.activated:
            if self.parent.activated and self.np_random.random() < self.activation_prob:
                self.activate()
        return super().step(action)
