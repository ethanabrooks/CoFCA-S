import unittest
from collections import defaultdict
from typing import Dict, TypeVar, Set, List

from string import ascii_lowercase
import numpy as np

X = TypeVar("X")
Graph = Dict[X, Dict[X, float]]


def dot_graph(graph: Graph):
    def generator():
        yield "Digraph {"
        for node, edges in graph.items():
            for adjacent, edge_length in edges.items():
                yield f"{node} -> {adjacent} [label={edge_length}]"
        yield "}"

    return "\n".join(generator())


def get_length(path: List[X], graph: Graph):
    if not path:
        return

    def get_edge_lengths():
        for a, b in zip(path, path[1:]):
            yield graph[a][b]

    return sum(get_edge_lengths())


def shortest_path(_from: X, _to: X, graph: Graph):
    explored = set()
    distances = defaultdict(lambda: np.inf)
    distances[_from] = 0
    prev = {_from: None}
    while True:
        if not distances:
            return None
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


def brute_force(_from: X, _to: X, graph: Graph):
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
    if not paths:
        return None
    minimum = min(paths, key=lambda p: sum(d for _, d in p))
    return [x for x, _ in minimum]


class TestDjikstra(unittest.TestCase):
    def test1(self):
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
        self.assertEqual(attempt, brute_force(_from="a", _to="d", graph=graph))
        self.assertEqual(attempt, ["a", "b", "c", "d",])

    def test2(self):
        """
        a --2-- b
        |       |
        1       1
        |       |
        c --3-- d
        """
        graph = dict(
            a=dict(b=2, c=1), b=dict(a=2, d=1), c=dict(a=1, d=3), d=dict(c=3, b=1)
        )
        attempt = shortest_path(_from="a", _to="d", graph=graph)
        self.assertEqual(attempt, brute_force(_from="a", _to="d", graph=graph))
        self.assertEqual(attempt, ["a", "b", "d"])

    def test_random(self):
        np.random.seed(0)
        alpha = ascii_lowercase[:10]
        for _ in range(100):
            size = np.random.randint(1, len(alpha))
            nodes = np.random.choice(list(alpha), replace=False, size=size)
            graph = {
                f: {
                    t: np.random.choice(10)
                    for t, c in zip(nodes, np.random.choice(2, size=size))
                    if c
                }
                for f in nodes
            }
            with self.subTest(graph=graph):
                _from, _to = np.random.choice(nodes, size=2)
                self.assertEqual(
                    get_length(
                        path=(shortest_path(_from=_from, _to=_to, graph=graph)),
                        graph=graph,
                    ),
                    get_length(
                        path=(brute_force(_from=_from, _to=_to, graph=graph)),
                        graph=graph,
                    ),
                    msg=dot_graph(graph),
                )


if __name__ == "__main__":
    unittest.main()
