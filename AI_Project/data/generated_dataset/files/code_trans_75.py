#!/usr/bin/python

"""Author: OMKAR PATHAK"""

from __future__ import annotations

from queue import Queue


class Graph:
    def __init__(self) -> None:
        self.vertices: dict[int, list[int]] = {}

    def print_graph(self) -> None:
        """
        prints adjacency list representation of graaph
        >>> x3 = Graph()
        >>> x3.print_graph()
        >>> x3.add_edge(0, 1)
        >>> x3.print_graph()
        0  :  1
        """
        for x4 in self.vertices:
            print(x4, " : ", " -> ".join([str(x5) for x5 in self.vertices[x4]]))

    def add_edge(self, x2: int, x7: int) -> None:
        """
        adding the edge between two vertices
        >>> x3 = Graph()
        >>> x3.print_graph()
        >>> x3.add_edge(0, 1)
        >>> x3.print_graph()
        0  :  1
        """
        if x2 in self.vertices:
            self.vertices[x2].append(x7)
        else:
            self.vertices[x2] = [x7]

    def bfs(self, x6: int) -> set[int]:
        """
        >>> x3 = Graph()
        >>> x3.add_edge(0, 1)
        >>> x3.add_edge(0, 1)
        >>> x3.add_edge(0, 2)
        >>> x3.add_edge(1, 2)
        >>> x3.add_edge(2, 0)
        >>> x3.add_edge(2, 3)
        >>> x3.add_edge(3, 3)
        >>> sorted(x3.bfs(2))
        [0, 1, 2, 3]
        """
        # initialize set for storing already x9 vertices
        x9 = set()

        # create a first in first out queue to store all the vertices for BFS
        queue: Queue = Queue()

        # mark the source node as x9 and enqueue it
        x9.add(x6)
        queue.put(x6)

        while not queue.empty():
            x8 = queue.get()

            # loop through all adjacent x8 and enqueue it if not yet x9
            for x1 in self.vertices[x8]:
                if x1 not in x9:
                    queue.put(x1)
                    x9.add(x1)
        return x9


if __name__ == "__main__":
    from doctest import testmod

    testmod(verbose=True)

    x3 = Graph()
    x3.add_edge(0, 1)
    x3.add_edge(0, 2)
    x3.add_edge(1, 2)
    x3.add_edge(2, 0)
    x3.add_edge(2, 3)
    x3.add_edge(3, 3)

    x3.print_graph()
    # 0  :  1 -> 2
    # 1  :  2
    # 2  :  0 -> 3
    # 3  :  3

    assert sorted(x3.bfs(2)) == [0, 1, 2, 3]
