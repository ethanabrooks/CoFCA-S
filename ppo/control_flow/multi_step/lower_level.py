from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple, Any
import numpy as np

from ppo.control_flow.multi_step.env import Env


@dataclass
class Item:
    pos: Tuple[int, int]
    type: Any


def act(src, dest, objects, inventory):
    def nearest(item):
        raise NotImplementedError

    def compute_direct_path(_src, _dest):
        yield _src
        _src = np.array(_src)
        while tuple(_src) != tuple(_dest):
            _src += np.clip(_dest - _src, -1, 1)
            yield tuple(_src)

    def obstructing(_path):
        return next(((o, p) for o, p in objects if p in set(_path)), (None, None))

    def adjacent(pos):
        raise NotImplementedError

    def alternate_paths(obstruction_pos):
        for pos in adjacent(obstruction_pos):
            if pos not in path | {obstruction_pos}:
                yield list(act(src, pos, objects, inventory)) + list(
                    act(pos, dest, objects, inventory)
                )

    *path, last2, last = compute_direct_path(src, dest)

    if last.obstruction == Env.wall:
        yield from min(alternate_paths(last2.pos), key=lambda p: len(p))

    elif last.obstruction == Env.water:
        if Env.wood in inventory:
            yield from act(src, last2.pos, objects, inventory)
        else:
            wood = nearest(Env.wood)
            if wood is None:
                yield
            yield from act(src, wood.pos, objects, inventory)
            yield Env.mine
            yield from act(wood.pos, last2.pos, objects - wood, inventory + Env.wood)
        yield Env.bridge
        bridge = Item(type=Env.bridge, pos=last.pos)
        yield from act(last2.pos, dest, objects + bridge, inventory - Env.wood)

    else:
        for step in path:
            yield step.action
        yield last.ation
