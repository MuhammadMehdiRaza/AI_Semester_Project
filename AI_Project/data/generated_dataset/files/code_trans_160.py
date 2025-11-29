"""
Problem Statement:
By starting at the top of the temp_8 below and moving to adjacent numbers on
the row below, the maximum total from top to bottom is 23.
3
7 4
2 4 6
8 5 9 3
That is, 3 + 7 + 4 + 9 = 23.
Find the maximum total from top to bottom in temp_8.txt (right click and
'Save Link/Target As...'), a 15K text file containing a temp_8 with
one-hundred rows.
"""

import os


def compute_1() -> int:
    """
    Finds the maximum total in a temp_8 as described by the problem statement
    above.
    >>> compute_1()
    7273
    """
    temp_7 = os.path.dirname(os.path.realpath(__file__))
    temp_9 = os.path.join(temp_7, "temp_8.txt")

    with open(temp_9) as temp_3:
        temp_8 = [[int(temp_2) for temp_2 in temp_6.split()] for temp_6 in temp_3]

    while len(temp_8) != 1:
        temp_5 = temp_8.pop()
        temp_1 = temp_8[-1]
        for temp_4 in range(len(temp_5) - 1):
            temp_1[temp_4] += max(temp_5[temp_4], temp_5[temp_4 + 1])
    return temp_8[0][0]


if __name__ == "__main__":
    print(compute_1())
