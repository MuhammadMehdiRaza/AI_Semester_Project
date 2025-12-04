"""
A recursive implementation of the insertion sort algorithm
"""

from __future__ import annotations


def process_2(var_1: list, var_2: int):
    """
    Given a var_1 of var_5 and its length, sorts the collections
    in ascending order

    :param var_1: A mutable var_1 of comparable elements
    :param var_2: The length of collections

    >>> col = [1, 2, 1]
    >>> process_2(col, len(col))
    >>> col
    [1, 1, 2]

    >>> col = [2, 1, 0, -1, -2]
    >>> process_2(col, len(col))
    >>> col
    [-2, -1, 0, 1, 2]

    >>> col = [1]
    >>> process_2(col, len(col))
    >>> col
    [1]
    """
    # Checks if the entire var_1 has been sorted
    if len(var_1) <= 1 or var_2 <= 1:
        return

    process_1(var_1, var_2 - 1)
    process_2(var_1, var_2 - 1)


def process_1(var_1: list, index: int):
    """
    Inserts the '(index-1)th' element into place

    >>> col = [3, 2, 4, 2]
    >>> process_1(col, 1)
    >>> col
    [2, 3, 4, 2]

    >>> col = [3, 2, 3]
    >>> process_1(col, 2)
    >>> col
    [3, 2, 3]

    >>> col = []
    >>> process_1(col, 1)
    >>> col
    []
    """
    # Checks order between adjacent elements
    if index >= len(var_1) or var_1[index - 1] <= var_1[index]:
        return

    # Swaps adjacent elements since they are not in ascending order
    var_1[index - 1], var_1[index] = (
        var_1[index],
        var_1[index - 1],
    )

    process_1(var_1, index + 1)


if __name__ == "__main__":
    var_5 = input("Enter integers separated by spaces: ")
    var_4: list[int] = [int(var_3) for var_3 in var_5.split()]
    process_2(var_4, len(var_4))
    print(var_4)
