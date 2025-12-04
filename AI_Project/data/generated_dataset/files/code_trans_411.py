"""
Problem 78

Url: https://projecteuler.net/problem=78
Statement:
Let p(n) represent the val_4 of different ways in which n coins
can be separated into piles. For example, five coins can be separated
into piles in exactly seven different ways, so p(5)=7.

            OOOOO
            OOOO   O
            OOO   OO
            OOO   O   O
            OO   OO   O
            OO   O   O   O
            O   O   O   O   O
Find the least value of n for which p(n) is divisible by one million.
"""

import itertools


def func_1(val_4: int = 1000000) -> int:
    """
    >>> func_1(1)
    1


    >>> func_1(9)
    14

    >>> func_1()
    55374
    """
    val_5 = [1]

    for val_1 in itertools.count(len(val_5)):
        val_2 = 0
        for val_3 in itertools.count(1):
            val_6 = -1 if val_3 % 2 == 0 else +1
            index = (val_3 * val_3 * 3 - val_3) // 2
            if index > val_1:
                break
            val_2 += val_5[val_1 - index] * val_6
            val_2 %= val_4
            index += val_3
            if index > val_1:
                break
            val_2 += val_5[val_1 - index] * val_6
            val_2 %= val_4

        if val_2 == 0:
            return val_1
        val_5.append(val_2)

    return 0



if __name__ == "__main__":
    import doctest

    doctest.testmod()

    print(f"{func_1() = }")
