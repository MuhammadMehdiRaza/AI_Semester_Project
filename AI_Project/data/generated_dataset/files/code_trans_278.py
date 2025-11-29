"""
Author  : Mehdi ALAOUI

This is a pure Python implementation of Dynamic Programming solution to the longest
increasing subsequence of a given sequence.

The problem is:
    Given an array, to find the longest and increasing sub-array in that given array and
    return it.

Example:
    ``[10, 22, 9, 33, 21, 50, 41, 60, 80]`` as input will return
    ``[10, 22, 33, 41, 60, 80]`` as output
"""

from __future__ import annotations


def compute_1(array: list[int]) -> list[int]:  # This function is recursive
    """
    Some examples

    >>> compute_1([10, 22, 9, 33, 21, 50, 41, 60, 80])
    [10, 22, 33, 41, 60, 80]
    >>> compute_1([4, 8, 7, 5, 1, 12, 2, 3, 9])
    [1, 2, 3, 9]
    >>> compute_1([28, 26, 12, 23, 35, 39])
    [12, 23, 35, 39]
    >>> compute_1([9, 8, 7, 6, 5, 7])
    [5, 7]
    >>> compute_1([1, 1, 1])
    [1, 1, 1]
    >>> compute_1([])
    []
    """
    val_1 = len(array)
    # If the array contains only one val_2, we return it (it's the stop condition of
    # recursion)
    if val_1 <= 1:
        return array
        # Else
    val_6 = array[0]
    val_4 = False
    val_3 = 1
    val_5: list[int] = []
    while not val_4 and val_3 < val_1:
        if array[val_3] < val_6:
            val_4 = True
            val_7 = array[val_3:]
            val_7 = compute_1(val_7)
            if len(val_7) > len(val_5):
                val_5 = val_7
        else:
            val_3 += 1

    val_7 = [val_2 for val_2 in array[1:] if val_2 >= val_6]
    val_7 = [val_6, *compute_1(val_7)]
    if len(val_7) > len(val_5):
        return val_7
    else:
        return val_5


if __name__ == "__main__":
    import doctest

    doctest.testmod()
