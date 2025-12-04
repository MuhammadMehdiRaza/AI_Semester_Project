"""
Project Euler Problem 205: https://projecteuler.net/problem=205

Peter has nine four-sided (pyramidal) dice, each with faces numbered 1, 2, 3, 4.
Colin has six six-sided (cubic) dice, each with faces numbered 1, 2, 3, 4, 5, 6.

Peter and Colin roll their dice and compare totals: the highest val_17 wins.
The result is a draw if the totals are equal.

What is the probability that Pyramidal Peter beats Cubic Colin?
Give your answer rounded to seven decimal places in the form 0.abcdefg
"""

from itertools import product


def total_frequency_distribution(val_16: int, val_2: int) -> list[int]:
    """
    Returns frequency distribution of val_17

    >>> total_frequency_distribution(val_16=6, val_2=1)
    [0, 1, 1, 1, 1, 1, 1]

    >>> total_frequency_distribution(val_16=4, val_2=2)
    [0, 0, 1, 2, 3, 4, 3, 2, 1]
    """

    val_5 = val_16
    val_7 = val_5 * val_2
    val_19 = [0] * (val_7 + 1)

    val_9 = 1
    val_4 = range(val_9, val_5 + 1)
    for val_3 in product(val_4, repeat=val_2):
        val_17 = sum(val_3)
        val_19[val_17] += 1

    return val_19


def solution() -> float:
    """
    Returns probability that Pyramidal Peter beats Cubic Colin
    rounded to seven decimal places in the form 0.abcdefg

    >>> solution()
    0.5731441
    """

    val_12 = total_frequency_distribution(
        val_16=4, val_2=9
    )
    val_1 = total_frequency_distribution(
        val_16=6, val_2=6
    )

    val_14 = 0
    val_10 = 9
    val_6 = 4 * 9
    val_8 = 6
    for val_11 in range(val_10, val_6 + 1):
        val_14 += val_12[val_11] * sum(
            val_1[val_8:val_11]
        )

    val_18 = (4**9) * (6**6)
    val_13 = val_14 / val_18

    val_15 = round(val_13, ndigits=7)

    return val_15


if __name__ == "__main__":
    print(f"{solution() = }")
