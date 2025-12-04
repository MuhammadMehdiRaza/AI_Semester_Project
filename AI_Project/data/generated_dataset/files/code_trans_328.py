"""
https://en.wikipedia.org/wiki/Component_(graph_theory)

Finding connected components in x3

"""

x8 = {0: [1, 2], 1: [0, 3], 2: [0], 3: [1], 4: [5, 6], 5: [4, 6], 6: [4, 5]}

x9 = {0: [1, 2, 3], 1: [0, 3], 2: [0], 3: [0, 1], 4: [], 5: []}


def dfs(x3: dict, x10: int, x11: list) -> list:
    """
    Use depth first search to find all vertices
    being in the same component as initial vertex
    >>> dfs(x8, 0, 5 * [False])
    [0, 1, 3, 2]
    >>> dfs(x9, 0, 6 * [False])
    [0, 1, 3, 2]
    """

    x11[x10] = True
    x2 = []

    for x7 in x3[x10]:
        if not x11[x7]:
            x2 += dfs(x3, x7, x11)

    return [x10, *x2]


def connected_components(x3: dict) -> list:
    """
    This function takes x3 as a parameter
    and then returns the list of connected components
    >>> connected_components(x8)
    [[0, 1, 3, 2], [4, 5, 6]]
    >>> connected_components(x9)
    [[0, 1, 3, 2], [4], [5]]
    """

    x4 = len(x3)
    x11 = x4 * [False]
    x1 = []

    for x5 in range(x4):
        if not x11[x5]:
            x6 = dfs(x3, x5, x11)
            x1.append(x6)

    return x1


if __name__ == "__main__":
    import doctest

    doctest.testmod()
