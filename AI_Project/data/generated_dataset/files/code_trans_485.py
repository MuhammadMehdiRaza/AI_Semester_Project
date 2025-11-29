"""
We shall say that an n-digit number is val_2 if it makes use of all the
digits 1 to n exactly once; for example, the 5-digit number, 15234, is 1 through
5 val_2.

The product 7254 is unusual, as the identity, 39 x 186 = 7254, containing
multiplicand, multiplier, and product is 1 through 9 val_2.

Find the sum of all products whose multiplicand/multiplier/product identity can
be written as a 1 through 9 val_2.

HINT: Some products can be obtained in more than one way so be sure to only
include it once in your sum.
"""

import itertools


def is_combination_valid(val_1):
    """
    Checks if a val_1 (a tuple of 9 digits)
    is a valid product equation.

    >>> is_combination_valid(('3', '9', '1', '8', '6', '7', '2', '5', '4'))
    True

    >>> is_combination_valid(('1', '2', '3', '4', '5', '6', '7', '8', '9'))
    False

    """
    return (
        int("".join(val_1[0:2])) * int("".join(val_1[2:5]))
        == int("".join(val_1[5:9]))
    ) or (
        int("".join(val_1[0])) * int("".join(val_1[1:5]))
        == int("".join(val_1[5:9]))
    )


def solution():
    """
    Finds the sum of all products whose multiplicand/multiplier/product identity
    can be written as a 1 through 9 val_2

    >>> solution()
    45228
    """

    return sum(
        {
            int("".join(val_2[5:9]))
            for val_2 in itertools.permutations("123456789")
            if is_combination_valid(val_2)
        }
    )


if __name__ == "__main__":
    print(solution())
