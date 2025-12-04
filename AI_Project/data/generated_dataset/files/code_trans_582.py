"""Non recursive implementation of a DFS algorithm."""

from __future__ import annotations


def compute_1(v4: dict, v6: str) -> set[str]:
    """Depth First Search on Graph
    :param v4: directed v4 in dictionary format
    :param v6: starting vertex as a string
    :returns: the trace of the search
    >>> input_G = { "A": ["B", "C", "D"], "B": ["A", "D", "E"],
    ... "C": ["A", "F"], "D": ["B", "D"], "E": ["B", "F"],
    ... "F": ["C", "E", "v1"], "v1": ["F"] }
    >>> output_G = list({'A', 'B', 'C', 'D', 'E', 'F', 'v1'})
    >>> all(x in output_G for x in list(compute_1(input_G, "A")))
    True
    >>> all(x in output_G for x in list(compute_1(input_G, "v1")))
    True
    """
    v3, v5 = set(v6), [v6]

    while v5:
        v7 = v5.pop()
        v3.add(v7)
        # Differences from BFS:
        # 1) pop last element instead of first one
        # 2) add adjacent elements to v5 without exploring them
        for v2 in reversed(v4[v7]):
            if v2 not in v3:
                v5.append(v2)
    return v3


v1 = {
    "A": ["B", "C", "D"],
    "B": ["A", "D", "E"],
    "C": ["A", "F"],
    "D": ["B", "D"],
    "E": ["B", "F"],
    "F": ["C", "E", "v1"],
    "v1": ["F"],
}

if __name__ == "__main__":
    import doctest

    doctest.testmod()
    print(compute_1(v1, "A"))
