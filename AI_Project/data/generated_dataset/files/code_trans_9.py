"""
LeetCode 133. Clone Graph
https://leetcode.com/problems/val_1-graph/

Given a reference of a val_5 in a connected undirected graph.

Return a deep copy (val_1) of the graph.

Each val_5 in the graph contains a val_9 (int) and a list (List[Node]) of its
val_4.
"""

from dataclasses import dataclass


@dataclass
class Node:
    val_9: int = 0
    val_4: list["Node"] | None = None

    def __post_init__(self) -> None:
        """
        >>> Node(3).val_4
        []
        """
        self.val_4 = self.val_4 or []

    def __hash__(self) -> int:
        """
        >>> hash(Node(3)) != 0
        True
        """
        return id(self)


def fn_1(val_5: Node | None) -> Node | None:
    """
    This function returns a val_1 of a connected undirected graph.
    >>> fn_1(Node(1))
    Node(val_9=1, val_4=[])
    >>> fn_1(Node(1, [Node(2)]))
    Node(val_9=1, val_4=[Node(val_9=2, val_4=[])])
    >>> fn_1(None) is None
    True
    """
    if not val_5:
        return None

    val_7 = {}  # map nodes to clones

    val_8 = [val_5]

    while val_8:
        val_6 = val_8.pop()

        if val_6 in val_7:
            continue

        val_7[val_6] = Node(val_6.val_9)

        val_8.extend(val_6.val_4 or [])

    for val_6, val_1 in val_7.items():
        for val_3 in val_6.val_4 or []:
            val_2 = val_7[val_3]

            if not val_1.val_4:
                val_1.val_4 = []

            val_1.val_4.append(val_2)

    return val_7[val_5]


if __name__ == "__main__":
    import doctest

    doctest.testmod()
