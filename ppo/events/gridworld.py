from collections import namedtuple

import gym
from gym.utils import seeding

import numpy as np

from ppo.events.objects import (
    Object,
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
)

State = namedtuple("State", "objects interactions")


class Gridworld(gym.Env):
    def __init__(
        self,
        cook_time: int,
        time_to_heat_oven: int,
        height: int,
        width: int,
        doorbell_prob: float,
        mouse_prob: float,
        baby_prob: float,
        mess_prob: float,
        fly_prob: float,
        toward_cat_prob: float,
    ):
        super().__init__()
        self.object_idxs = {}
        self.height = height
        self.width = width
        self.random, seed = seeding.np_random(0)
        self.grasping = None
        multiple_object_types = [Mess, Fly]
        self.object_types = object_types = [
            Agent,
            Mouse,
            Dog,
            Baby,
            Door,
            MouseHole,
            Table,
            Oven,
            Food,
            Cat,
            Mess,
            Fire,
            Mess,
            Fly,
        ]

        def make_objects():
            objects = []
            for object_type in object_types:
                kwargs = dict(
                    objects=objects,
                    object_types=object_types,
                    height=height,
                    width=width,
                    random=self.random,
                )
                if object_type is Door:
                    kwargs.update(activation_prob=doorbell_prob)
                if object_type is Mouse:
                    kwargs.update(activation_prob=mouse_prob)
                if object_type is Baby:
                    kwargs.update(activation_prob=baby_prob)
                if object_type is Mess:
                    kwargs.update(activation_prob=mess_prob)
                if object_type is Fly:
                    kwargs.update(activation_prob=fly_prob)
                if object_type is Dog:
                    kwargs.update(toward_cat_prob=toward_cat_prob)
                if object_type is Food:
                    kwargs.update(cook_time=cook_time)
                if object_type is Oven:
                    kwargs.update(time_to_heat=time_to_heat_oven)
                if object_type in multiple_object_types:
                    for _ in range(height * width):
                        objects += [object_type(**kwargs)]
                else:
                    objects += [object_type(**kwargs)]
            return objects

        self.objects = None
        self.make_objects = make_objects
        self.agent = None
        self.last_action = None
        self.transitions = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        self._mess = None

    def interact(self, i):
        object_type = self.object_types[i]
        for obj in self.objects:
            if type(obj) is object_type and obj.pos == self.agent.pos:
                if self.grasping is None:
                    obj.interact()
                    self.grasping = obj if obj.grasped else None
                else:
                    self.grasping.interact()
                    self.grasping = None
                yield obj

    def step(self, a):
        a = int(a)
        self.last_action = a
        n_transitions = len(self.transitions)
        if a < n_transitions:
            action = self.transitions[int(a)]
            interactions = []
        else:
            action = (0, 0)
            interactions = list(self.interact(a - n_transitions))

        obj: Object
        for obj in self.objects:
            obj.step(action)
        grasping = self.agent.grasping
        if grasping:
            try:
                grasping.set_pos(self.agent.pos)
            except AttributeError:
                pass

        return State(objects=self.objects, interactions=interactions), 0, False, {}

    def seed(self, seed=None):
        self.random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.objects = self.make_objects()
        self.agent = next((o for o in self.objects if type(o) is Agent))
        return State(objects=self.objects, interactions=[])

    def render(self, mode="human"):
        pass
