"""
Project Euler Problem 1: https://projecteuler.net/problem=1

Multiples of 3 and 5

If we list all the natural numbers below 10 that are multiples of 3 or 5,
we get 3, 5, 6 and 9. The sum of these multiples is 23.

Find the sum of all the multiples of 3 or 5 below 1000.
"""


def compute_1(x2: int = 1000) -> int:
    """
    Returns the sum of all the multiples of 3 or 5 below x2.

    >>> compute_1(3)
    0
    >>> compute_1(4)
    3
    >>> compute_1(10)
    23
    >>> compute_1(600)
    83700
    """

    x1 = 3
    x3 = 0
    while x1 < x2:
        if x1 % 3 == 0 or x1 % 5 == 0:
            x3 += x1
        elif x1 % 15 == 0:
            x3 -= x1
        x1 += 1
    return x3


if __name__ == "__main__":
    print(f"{compute_1() = }")
