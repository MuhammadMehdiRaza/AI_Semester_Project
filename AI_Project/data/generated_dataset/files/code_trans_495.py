"""
Problem 13: https://projecteuler.net/problem=13

Problem Statement:
Work out the first ten digits of the sum of the following one-hundred 50-digit
numbers.
"""

import os


def process_1():
    """
    Returns the first ten digits of the sum of the array elements
    from the file num.txt

    >>> process_1()
    '5537376230'
    """
    x2 = os.path.join(os.path.dirname(__file__), "num.txt")
    with open(x2) as x1:
        return str(sum(int(x3) for x3 in x1))[:10]


if __name__ == "__main__":
    print(process_1())
