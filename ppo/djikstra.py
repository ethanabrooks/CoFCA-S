from collections import defaultdict
from typing import Dict, Any, TypeVar, Set

import numpy as np

X = TypeVar("X")


def shortest_path(_from: X, _to: X, graph: Dict[X, Dict[X, float]]):
    explored = set()
    distances = defaultdict(lambda: np.inf)
    distances[_from] = 0
    prev = {_from: None}
    while True:
        distance, value = min(
            [(d, v) for v, d in distances.items()]
        )  # TODO: not efficient
        if value == _to:  # done (_to is minimum distance)

            def generator(v):
                while v is not None:
                    yield v
                    v = prev[v]

            return list(reversed(list(generator(value))))

        explored.add(value)
        del distances[value]
        for adjacent, edge_length in graph[value].items():
            if (
                adjacent not in explored
                and distance + edge_length < distances[adjacent]
            ):
                distances[adjacent] = distance + edge_length
                prev[adjacent] = value


def brute_force(_from: X, _to: X, graph: Dict[X, Dict[X, float]]):
    def get_paths(f: X, explored: Set[X]):
        if f == _to:
            yield [(f, 0)]
        else:
            explored = explored | {f}
            for adjacent, edge_length in graph[f].items():
                if adjacent not in explored:
                    for path in get_paths(adjacent, explored):
                        yield [(f, edge_length)] + path

    paths = list(get_paths(_from, explored=set()))
    minimum = min(paths, key=lambda p: sum(d for _, d in p))
    return [x for x, _ in minimum]


def main():
    """
  0_
  |_a --1-- b
    |    __/|
    3 __1   3
    |/      |
    c --1-- d
    """

    graph = dict(
        a=dict(b=1, c=3, a=0),
        b=dict(a=1, d=3, c=1),
        c=dict(a=3, b=1, d=1),
        d=dict(c=1, b=3),
    )
    attempt = shortest_path(_from="a", _to="d", graph=graph)
    assert (
        attempt == brute_force(_from="a", _to="d", graph=graph) == ["a", "b", "c", "d",]
    )

    """
    a --2-- b
    |       |
    1       1
    |       |
    c --3-- d
    """
    graph = dict(a=dict(b=2, c=1), b=dict(a=2, d=1), c=dict(a=1, d=3), d=dict(c=3, b=1))
    attempt = shortest_path(_from="a", _to="d", graph=graph)
    assert attempt == brute_force(_from="a", _to="d", graph=graph) == ["a", "b", "d"]


if __name__ == "__main__":
    main()
