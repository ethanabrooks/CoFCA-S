from enum import unique, Enum, auto

from colored import fg

from utils import RESET


@unique
class Terrain(Enum):
    FACTORY = auto()
    WATER = auto()
    MOUNTAIN = auto()
    WALL = auto()


@unique
class Other(Enum):
    AGENT = auto()
    MAP = auto()


@unique
class Resource(Enum):
    WOOD = auto()
    STONE = auto()
    IRON = auto()


Symbols = {
    Resource.WOOD: fg("green") + "w",
    Resource.STONE: fg("light_slate_grey") + "s",
    Resource.IRON: fg("grey_100") + "i",
    Terrain.FACTORY: fg("red") + "f",
    Terrain.WALL: RESET + "#",
    Terrain.WATER: fg("blue") + "~",
    Terrain.MOUNTAIN: fg("sandy_brown") + "M",
    Other.AGENT: fg("yellow") + "A",
}
Refined = Enum(value="Refined", names=[x.name for x in Resource])
InventoryItems = [Other.MAP] + list(Resource) + list(Refined)
WorldObjects = [Other.AGENT] + list(Resource) + list(Terrain)
Necessary = list(Resource) + [Terrain.FACTORY]


@unique
class Interaction(Enum):
    COLLECT = auto()
    REFINE = auto()
    CROSS = auto()
