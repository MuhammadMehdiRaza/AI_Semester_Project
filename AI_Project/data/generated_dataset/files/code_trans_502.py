"""
v9 function takes a list of sorted integer lists and finds the smallest
range that includes at least one number from each list, using a min heap for efficiency.
"""

from heapq import heappop, heappush
from sys import maxsize


def v9(v8: list[list[int]]) -> list[int]:
    """
    Find the smallest range from each list in v8.

    Uses min heap for efficiency. The range includes at least one number from each list.

    Args:
        `v8`: List of k sorted integer lists.

    Returns:
        list: Smallest range as a two-element list.

    Examples:

    >>> v9([[4, 10, 15, 24, 26], [0, 9, 12, 20], [5, 18, 22, 30]])
    [20, 24]
    >>> v9([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    [1, 1]
    >>> v9(((1, 2, 3), (1, 2, 3), (1, 2, 3)))
    [1, 1]
    >>> v9(((-3, -2, -1), (0, 0, 0), (1, 2, 3)))
    [-1, 1]
    >>> v9([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    [3, 7]
    >>> v9([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    [0, 0]
    >>> v9([[], [], []])
    Traceback (most recent call last):
        ...
    IndexError: list index out of range
    """

    v6: list[tuple[int, int, int]] = []
    v1 = -maxsize - 1

    for v4, items in enumerate(v8):
        heappush(v6, (items[0], v4, 0))
        v1 = max(v1, items[0])

    # Initialize v9 with large integer values
    v9 = [-maxsize - 1, maxsize]

    while v6:
        v2, v5, v3 = heappop(v6)

        if v1 - v2 < v9[1] - v9[0]:
            v9 = [v2, v1]

        if v3 == len(v8[v5]) - 1:
            break

        v7 = v8[v5][v3 + 1]
        heappush(v6, (v7, v5, v3 + 1))
        v1 = max(v1, v7)

    return v9


if __name__ == "__main__":
    from doctest import testmod

    testmod()
    print(f"{v9([[1, 2, 3], [1, 2, 3], [1, 2, 3]])}")  # Output: [1, 1]
