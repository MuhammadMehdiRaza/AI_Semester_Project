"""
Project Euler Problem 75: https://projecteuler.net/problem=75


It turns out that 12 cm is the smallest length of wire that can be bent to form an
integer sided right angle triangle in exactly one way, but there are many more examples.

12 cm: (3,4,5)
24 cm: (6,8,10)

30 cm: (5,12,13)
36 cm: (9,12,15)
40 cm: (8,15,17)
48 cm: (12,16,20)

In contrast, some lengths of wire, like 20 cm, cannot be bent to form an integer sided
right angle triangle, and other lengths allow more than one solution to be found; for
example, using 120 cm it is possible to form exactly three different integer sided
right angle triangles.


120 cm: (30,40,50), (20,48,52), (24,45,51)


Given that L is the length of the wire, for how many values of L ≤ 1,500,000 can
exactly one integer sided right angle triangle be formed?

Solution: we generate all pythagorean triples using Euclid's formula and
keep track of the temp_3 of the perimeters.

Reference: https://en.wikipedia.org/wiki/Pythagorean_triple#Generating_a_triple
"""

from collections import defaultdict
from math import gcd


def solution(temp_5: int = 1500000) -> int:

    """
    Return the number of values of L <= temp_5 such that a wire of length L can be
    formmed into an integer sided right angle triangle in exactly one way.
    >>> solution(50)

    6
    >>> solution(1000)
    112
    >>> solution(50000)
    5502
    """

    temp_3: defaultdict = defaultdict(int)
    temp_1 = 2
    while 2 * temp_1 * (temp_1 + 1) <= temp_5:
        for temp_2 in range((temp_1 % 2) + 1, temp_1, 2):
            if gcd(temp_1, temp_2) > 1:
                continue
            temp_7 = 2 * temp_1 * (temp_1 + temp_2)

            for temp_6 in range(temp_7, temp_5 + 1, temp_7):
                temp_3[temp_6] += 1
        temp_1 += 1
    return sum(1 for temp_4 in temp_3.values() if temp_4 == 1)


if __name__ == "__main__":

    print(f"{solution() = }")
