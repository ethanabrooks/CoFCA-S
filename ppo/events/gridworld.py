import functools
from collections import namedtuple
import numpy as np

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


class Gridworld(gym.Env):
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
        self.height = height
        self.width = width
        multiple_object_types = [Mess, Fly]
        object_types = [
            Door,
            MouseHole,
            Mouse,
            Baby,
            Refrigerator,
            Table,
            Oven,
            Food,
            Dog,
            Cat,
            Mess,
            Fire,
            Mess,
            Fly,
            Agent,
        ]
        assert object_types[-1] is Agent
        self.objects = []
        for object_type in object_types:
            kwargs = dict(objects=self.objects, height=height, width=width)
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

        self.agent = None
        self.last_action = None
        self.transitions = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]

    def interact(self):
        if self.agent.grasping is None:
            for obj in self.objects:
                if obj.pos == self.agent.pos:
                    obj.interact()
                    yield obj
                    if isinstance(obj, Graspable):
                        self.agent.grasp(obj)
        else:
            self.agent.grasping.interact()
            self.agent.grasping = None

    def step(self, a):
        self.last_action = a
        action = self.transitions[a]
        interactions = list(self.interact()) if action == (0, 0) else []

        obj: Object
        for obj in self.objects:
            obj.step(action)
        grasping = self.agent.grasping
        if grasping:
            assert isinstance(grasping, Graspable)
            grasping.set_pos(self.agent.pos)

        return State(objects=self.objects, interactions=interactions), 0, False, {}

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        for obj in self.objects:
            obj.reset()
            if isinstance(obj, Agent):
                self.agent = obj
        return State(objects=self.objects, interactions=[])

    def render(self, mode="human"):
        pass
