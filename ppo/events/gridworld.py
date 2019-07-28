import functools
from collections import namedtuple

import gym

from ppo.events.objects import (
    Object,
    Graspable,
    Agent,
    Door,
    MouseHole,
    Mouse,
    Baby,
    Oven,
    Refrigerator,
    Table,
    Food,
    Dog,
    Cat,
    Mess,
    Fire,
    Fly,
    RandomActivating,
)

State = namedtuple("State", "objects interactions")


class EventsGridworld(gym.Env):
    def __init__(
        self,
        cook_time: int,
        time_to_heat_oven: int,
        random_activation_prob: float,
        height: int,
        width: int,
    ):
        super().__init__()
        self.object_idxs = {}
        single_object_types = [
            Agent,
            Door,
            MouseHole,
            Mouse,
            Baby,
            Refrigerator,
            Table,
            Food,
            Oven,
            Dog,
            Cat,
            Mess,
            Fire,
            Mess,
            Fly,
        ]
        multiple_object_types = [Mess, Fly]
        self.objects = []
        for object_type in single_object_types:
            kwargs = dict(
                random=self.np_random, objects=self.objects, height=height, width=width
            )
            if issubclass(object_type, RandomActivating):
                kwargs.update(activation_prob=random_activation_prob)
            if object_type is Food:
                kwargs.update(cook_time=cook_time)
            if object_type is Oven:
                kwargs.update(time_to_heat=time_to_heat_oven)
            if object_type in multiple_object_types:
                for _ in range(height * width):
                    self.objects.append(object_type(**kwargs))
            else:
                self.objects.append(object_type(**kwargs))

        self.grasping = None
        self.agent = None
        self.last_action = None
        self.transitions = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]

    def step(self, a):
        self.last_action = a
        action = self.transitions[a]
        interacting = a == 0
        interactions = []

        obj: Object
        for obj in self.objects:
            obj.step(action)
            if interacting and obj.pos == self.agent.pos:
                interactions.append(obj)
                obj.interact()
                if isinstance(obj, Graspable):
                    if self.grasping is obj:
                        self.grasping = None
                    elif self.grasping is None:
                        self.grasping = obj

        return State(objects=self.objects, interactions=interactions), 0, False, {}

    def reset(self):
        for obj in self.objects:
            obj.reset()
        return State(objects=self.objects, interactions=[])

    def render(self, mode="human"):
        pass
