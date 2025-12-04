#!/usr/bin/env python3

"""Provide the functionality to manipulate a single bit."""


def compute_5(temp_1: int, temp_2: int) -> int:
    """
    Set the bit at temp_2 to 1.

    Details: perform bitwise or for given temp_1 and X.
    Where X is a temp_1 with all the bits - zeroes and bit on given
    temp_2 - one.

    >>> compute_5(0b1101, 1) # 0b1111
    15
    >>> compute_5(0b0, 5) # 0b100000
    32
    >>> compute_5(0b1111, 1) # 0b1111
    15
    """
    return temp_1 | (1 << temp_2)


def compute_1(temp_1: int, temp_2: int) -> int:
    """
    Set the bit at temp_2 to 0.

    Details: perform bitwise and for given temp_1 and X.
    Where X is a temp_1 with all the bits - ones and bit on given
    temp_2 - zero.

    >>> compute_1(0b10010, 1) # 0b10000
    16
    >>> compute_1(0b0, 5) # 0b0
    0
    """
    return temp_1 & ~(1 << temp_2)


def compute_2(temp_1: int, temp_2: int) -> int:
    """
    Flip the bit at temp_2.

    Details: perform bitwise xor for given temp_1 and X.
    Where X is a temp_1 with all the bits - zeroes and bit on given
    temp_2 - one.

    >>> compute_2(0b101, 1) # 0b111
    7
    >>> compute_2(0b101, 0) # 0b100
    4
    """
    return temp_1 ^ (1 << temp_2)


def compute_4(temp_1: int, temp_2: int) -> bool:
    """
    Is the bit at temp_2 set?

    Details: Shift the bit at temp_2 to be the first (smallest) bit.
    Then check if the first bit is set by anding the shifted temp_1 with 1.

    >>> compute_4(0b1010, 0)
    False
    >>> compute_4(0b1010, 1)
    True
    >>> compute_4(0b1010, 2)
    False
    >>> compute_4(0b1010, 3)
    True
    >>> compute_4(0b0, 17)
    False
    """
    return ((temp_1 >> temp_2) & 1) == 1


def compute_3(temp_1: int, temp_2: int) -> int:
    """
    Get the bit at the given temp_2

    Details: perform bitwise and for the given temp_1 and X,
    Where X is a temp_1 with all the bits - zeroes and bit on given temp_2 - one.
    If the result is not equal to 0, then the bit on the given temp_2 is 1, else 0.

    >>> compute_3(0b1010, 0)
    0
    >>> compute_3(0b1010, 1)
    1
    >>> compute_3(0b1010, 2)
    0
    >>> compute_3(0b1010, 3)
    1
    """
    return int((temp_1 & (1 << temp_2)) != 0)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
