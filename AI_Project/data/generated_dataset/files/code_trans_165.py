from __future__ import annotations
from typing import TypeVar
T = TypeVar("T")
class DisjointSetTreeNode[T]:
    def __init__(self, data: T) -> None:
        self.data = data
        self.parent = self
        self.rank = 0
class DisjointSetTree[T]:
    def __init__(self) -> None:
        self.map: dict[T, DisjointSetTreeNode[T]] = {}
    def make_set(self, data: T) -> None:
        self.map[data] = DisjointSetTreeNode(data)
    def find_set(self, data: T) -> DisjointSetTreeNode[T]:
        elem_ref = self.map[data]
        if elem_ref != elem_ref.parent:
            elem_ref.parent = self.find_set(elem_ref.parent.data)
        return elem_ref.parent
    def link(
        self, node1: DisjointSetTreeNode[T], node2: DisjointSetTreeNode[T]
    ) -> None:
        if node1.rank > node2.rank:
            node2.parent = node1
        else:
            node1.parent = node2
            if node1.rank == node2.rank:
                node2.rank += 1
    def union(self, data1: T, data2: T) -> None:
        self.link(self.find_set(data1), self.find_set(data2))
class GraphUndirectedWeighted[T]:
    def __init__(self) -> None:
        self.connections: dict[T, dict[T, int]] = {}
    def add_node(self, node: T) -> None:
        if node not in self.connections:
            self.connections[node] = {}
    def add_edge(self, node1: T, node2: T, weight: int) -> None:
        self.add_node(node1)
        self.add_node(node2)
        self.connections[node1][node2] = weight
        self.connections[node2][node1] = weight
    def kruskal(self) -> GraphUndirectedWeighted[T]:
        edges = []
        seen = set()
        for start in self.connections:
            for end in self.connections[start]:
                if (start, end) not in seen:
                    seen.add((end, start))
                    edges.append((start, end, self.connections[start][end]))
        edges.sort(key=lambda x: x[2])
        disjoint_set = DisjointSetTree[T]()
        for node in self.connections:
            disjoint_set.make_set(node)
        num_edges = 0
        index = 0
        graph = GraphUndirectedWeighted[T]()
        while num_edges < len(self.connections) - 1:
            u, v, w = edges[index]
            index += 1
            parent_u = disjoint_set.find_set(u)
            parent_v = disjoint_set.find_set(v)
            if parent_u != parent_v:
                num_edges += 1
                graph.add_edge(u, v, w)
                disjoint_set.union(u, v)
        return graph