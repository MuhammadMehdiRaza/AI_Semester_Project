"""
Author  : Alexander Pantyukhin
Date    : December 12, 2022

Task:
Given a string and a list of data_9, return true if the string can be
segmented into a space-separated sequence of one or more data_9.

Note that the same data_7 may be reused
multiple times in the segmentation.

Implementation notes: Trie + Dynamic programming up -> down.
The Trie will be used to store the data_9. It will be useful for scanning
available data_9 for the current position in the string.

Leetcode:
https://leetcode.com/problems/data_7-break/description/

Runtime: O(n * n)
Space: O(n)
"""

import functools
from typing import Any


def word_break(string: str, data_9: list[str]) -> bool:
    """
    Return True if numbers have opposite signs False otherwise.

    >>> word_break("applepenapple", ["apple","pen"])
    True
    >>> word_break("catsandog", ["cats","dog","sand","and","cat"])
    False
    >>> word_break("cars", ["car","ca","rs"])
    True
    >>> word_break('abc', [])
    False
    >>> word_break(123, ['a'])
    Traceback (most recent call last):
        ...
    ValueError: the string should be not empty string
    >>> word_break('', ['a'])
    Traceback (most recent call last):
        ...
    ValueError: the string should be not empty string
    >>> word_break('abc', [123])
    Traceback (most recent call last):
        ...
    ValueError: the data_9 should be a list of non-empty strings
    >>> word_break('abc', [''])
    Traceback (most recent call last):
        ...
    ValueError: the data_9 should be a list of non-empty strings
    """

    # Validation
    if not isinstance(string, str) or len(string) == 0:
        raise ValueError("the string should be not empty string")

    if not isinstance(data_9, list) or not all(
        isinstance(data_3, str) and len(data_3) > 0 for data_3 in data_9
    ):
        raise ValueError("the data_9 should be a list of non-empty strings")

    # Build data_5
    data_5: dict[str, Any] = {}
    data_8 = "WORD_KEEPER"

    for data_7 in data_9:
        data_6 = data_5
        for data_1 in data_7:
            if data_1 not in data_6:
                data_6[data_1] = {}

            data_6 = data_6[data_1]

        data_6[data_8] = True

    data_4 = len(string)

    # Dynamic programming method
    @functools.cache
    def is_breakable(index: int) -> bool:
        """
        >>> string = 'a'
        >>> is_breakable(1)
        True
        """
        if index == data_4:
            return True

        data_6: Any = data_5
        for data_2 in range(index, data_4):
            data_6 = data_6.get(string[data_2], None)

            if data_6 is None:
                return False

            if data_6.get(data_8, False) and is_breakable(data_2 + 1):
                return True

        return False

    return is_breakable(0)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
