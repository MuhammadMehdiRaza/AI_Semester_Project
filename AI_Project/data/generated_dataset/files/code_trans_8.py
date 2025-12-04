"""
Author  : Sanjay Muthu <https://github.com/XenoBytesX>

This is a pure Python implementation of Dynamic Programming solution to the longest
increasing subsequence of a given sequence.

The problem is:
    Given an array, to find the longest and increasing sub-array in that given array and
    return it.

Example:
    ``[10, 22, 9, 33, 21, 50, 41, 60, 80]`` as input will return
    ``[10, 22, 33, 50, 60, 80]`` as output
"""

from __future__ import annotations

import copy


def longest_subsequence(array: list[int]) -> list[int]:
    """
    Some examples

    >>> longest_subsequence([10, 22, 9, 33, 21, 50, 41, 60, 80])
    [10, 22, 33, 50, 60, 80]
    >>> longest_subsequence([4, 8, 7, 5, 1, 12, 2, 3, 9])
    [1, 2, 3, 9]
    >>> longest_subsequence([9, 8, 7, 6, 5, 7])
    [7, 7]
    >>> longest_subsequence([28, 26, 12, 23, 35, 39])
    [12, 23, 35, 39]
    >>> longest_subsequence([1, 1, 1])
    [1, 1, 1]
    >>> longest_subsequence([])
    []
    """
    data_3 = len(array)
    # The longest increasing subsequence ending at array[data_1]
    data_2 = []
    for data_1 in range(data_3):
        data_2.append([array[data_1]])

    for data_1 in range(1, data_3):
        for data_4 in range(data_1):
            # If array[data_4] is less than or equal to array[data_1], then
            # data_2[data_4] + array[data_1]
            # is a valid increasing subsequence

            # data_2[data_1] is only set to
            # data_2[data_4] + array[data_1] if the length is longer.

            if array[data_4] <= array[data_1] and len(
                data_2[data_4]
            ) + 1 > len(data_2[data_1]):
                data_2[data_1] = copy.copy(
                    data_2[data_4]
                )
                data_2[data_1].append(array[data_1])

    data_5: list[int] = []
    for data_1 in range(data_3):
        if len(data_2[data_1]) > len(data_5):
            data_5 = data_2[data_1]

    return data_5


if __name__ == "__main__":
    import doctest

    doctest.testmod()
