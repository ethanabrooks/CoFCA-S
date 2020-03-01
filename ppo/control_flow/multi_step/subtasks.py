from attr import attr
from attr.validators import in_, instance_of
from sumtypes import sumtype, constructor


@attr.s
class Coord(object):
    x = attr.ib(validator=in_(list(range(6))))
    y = attr.ib(validator=in_(list(range(6))))


@sumtype
class Item:
    Wood = constructor()
    Gold = constructor()
    Iron = constructor()


@sumtype
class Terrain:
    Bridge = constructor()
    River = constructor()
    Merchant = constructor()
    Wall = constructor()
    Flat = constructor()


@sumtype
class Subtask:
    Mine = constructor(item=attr.ib(validator=instance_of(Item)))
    GoTo = constructor(coord=attr.ib(validator=instance_of(Coord)))
    Place = constructor(
        item=attr.ib(validator=instance_of(Item)),
        coord=attr.ib(validator=instance_of(Coord)),
    )
    BuildBridge = constructor()
    Sell = constructor(iterm=attr.ib(validator=instance_of(Item)))
