import heapq
from collections import deque, defaultdict
from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Set, Dict, Iterable


@dataclass(order=True)
class Node:
    distance: float
    id: int = field(compare=False)


def djikstra(start, dest, graph: Dict[int, Iterable[Node]]):
    nodes = {i: Node(id=i, distance=float("inf")) for i in graph}
    explored = set()  # type: Set[int]
    exploring = [Node(id=start, distance=0)]

    while True:
        node = heappop(exploring)
        if node.id == dest:
            return node.distance
        explored.add(node.id)
        new = {n.id for n in graph[node.id]} - explored
        for edge in new:
            node = nodes[edge.id]  # this is the actual node that may be in the heap
            node.distance = min(node.distance, node.distance + edge.distance)
            heappush(exploring, node)
