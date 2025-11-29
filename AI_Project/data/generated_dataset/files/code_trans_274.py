from __future__ import annotations

from typing import Any


class ContainsLoopError(Exception):
    pass


class Node:
    def __init__(self, temp_1: Any) -> None:
        self.temp_1: Any = temp_1
        self.next_node: Node | None = None

    def __iter__(self):
        temp_2 = self
        temp_4 = set()
        while temp_2:
            if temp_2 in temp_4:
                raise ContainsLoopError
            temp_4.add(temp_2)
            yield temp_2.temp_1
            temp_2 = temp_2.next_node

    @property
    def compute_1(self) -> bool:
        """
        A loop is when the exact same Node appears more than once in a linked list.
        >>> temp_3 = Node(1)
        >>> temp_3.next_node = Node(2)
        >>> temp_3.next_node.next_node = Node(3)
        >>> temp_3.next_node.next_node.next_node = Node(4)
        >>> temp_3.compute_1
        False
        >>> temp_3.next_node.next_node.next_node = temp_3.next_node
        >>> temp_3.compute_1
        True
        """
        try:
            list(self)
            return False
        except ContainsLoopError:
            return True


if __name__ == "__main__":
    temp_3 = Node(1)
    temp_3.next_node = Node(2)
    temp_3.next_node.next_node = Node(3)
    temp_3.next_node.next_node.next_node = Node(4)
    print(temp_3.compute_1)  # False
    temp_3.next_node.next_node.next_node = temp_3.next_node
    print(temp_3.compute_1)  # True

    temp_3 = Node(5)
    temp_3.next_node = Node(6)
    temp_3.next_node.next_node = Node(5)
    temp_3.next_node.next_node.next_node = Node(6)
    print(temp_3.compute_1)  # False

    temp_3 = Node(1)
    print(temp_3.compute_1)  # False
