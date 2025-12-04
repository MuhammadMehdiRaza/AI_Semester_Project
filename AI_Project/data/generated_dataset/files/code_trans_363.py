# Knight Tour Intro: https://www.youtube.com/watch?v=ab_dY3dZFHM

from __future__ import annotations


def get_valid_pos(data_11: tuple[int, int], data_8: int) -> list[tuple[int, int]]:
    """
    Find all the valid data_12 a knight can move to from the current data_11.

    >>> get_valid_pos((1, 3), 4)
    [(2, 1), (0, 1), (3, 2)]
    """

    data_16, data_14 = data_11
    data_12 = [
        (data_16 + 1, data_14 + 2),
        (data_16 - 1, data_14 + 2),
        (data_16 + 1, data_14 - 2),
        (data_16 - 1, data_14 - 2),
        (data_16 + 2, data_14 + 1),
        (data_16 + 2, data_14 - 1),
        (data_16 - 2, data_14 + 1),
        (data_16 - 2, data_14 - 1),
    ]
    data_9 = []

    for data_5 in data_12:
        data_17, data_15 = data_5
        if 0 <= data_17 < data_8 and 0 <= data_15 < data_8:
            data_9.append(data_5)

    return data_9


def is_complete(data_1: list[list[int]]) -> bool:
    """
    Check if the data_1 (matrix) has been completely filled with non-zero values.

    >>> is_complete([[1]])
    True

    >>> is_complete([[1, 2], [3, 0]])
    False
    """

    return not any(data_3 == 0 for data_13 in data_1 for data_3 in data_13)


def open_knight_tour_helper(
    data_1: list[list[int]], data_10: tuple[int, int], data_2: int
) -> bool:
    """
    Helper function to solve knight tour problem.
    """

    if is_complete(data_1):
        return True

    for data_11 in get_valid_pos(data_10, len(data_1)):
        data_16, data_14 = data_11

        if data_1[data_16][data_14] == 0:
            data_1[data_16][data_14] = data_2 + 1
            if open_knight_tour_helper(data_1, data_11, data_2 + 1):
                return True
            data_1[data_16][data_14] = 0

    return False


def open_knight_tour(data_8: int) -> list[list[int]]:
    """
    Find the solution for the knight tour problem for a data_1 of size data_8. Raises
    ValueError if the tour cannot be performed for the given size.

    >>> open_knight_tour(1)
    [[1]]

    >>> open_knight_tour(2)
    Traceback (most recent call last):
        ...
    ValueError: Open Knight Tour cannot be performed on a data_1 of size 2
    """

    data_1 = [[0 for data_4 in range(data_8)] for data_6 in range(data_8)]

    for data_4 in range(data_8):
        for data_6 in range(data_8):
            data_1[data_4][data_6] = 1
            if open_knight_tour_helper(data_1, (data_4, data_6), 1):
                return data_1
            data_1[data_4][data_6] = 0

    data_7 = f"Open Knight Tour cannot be performed on a data_1 of size {data_8}"
    raise ValueError(data_7)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
