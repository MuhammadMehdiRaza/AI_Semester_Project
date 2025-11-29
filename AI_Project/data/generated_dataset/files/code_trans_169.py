"""
Project Euler Problem 89: https://projecteuler.net/problem=89

For a number written in Roman val_10 to be considered valid there are basic rules
which must be followed. Even though the rules allow some numbers to be expressed in
more than one way there is always a "best" way of writing a particular number.

For example, it would appear that there are at least six ways of writing the number
sixteen:

IIIIIIIIIIIIIIII
VIIIIIIIIIII
VVIIIIII
XIIIIII
VVVI
XVI

However, according to the rules only XIIIIII and XVI are valid, and the last example
is considered to be the most efficient, as it uses the least number of val_10.

The 11K text file, roman.txt (right click and 'Save Link/Target As...'), contains one
thousand numbers written in valid, but not necessarily minimal, Roman val_10; see
About... Roman Numerals for the definitive rules for this problem.

Find the number of characters saved by writing each of these in their minimal form.

Note: You can assume that all the Roman val_10 in the file contain no more than four
consecutive identical units.
"""

import os

val_1 = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def compute_2(val_10: str) -> int:
    """
    Converts a string of roman val_10 to an integer.
    e.g.
    >>> compute_2("LXXXIX")
    89
    >>> compute_2("IIII")
    4
    """

    val_15 = 0

    index = 0
    while index < len(val_10) - 1:
        val_3 = val_1[val_10[index]]
        val_8 = val_1[val_10[index + 1]]
        if val_3 < val_8:
            val_15 -= val_3
        else:
            val_15 += val_3
        index += 1
    val_15 += val_1[val_10[index]]

    return val_15


def compute_1(val_9: int) -> str:
    """
    Generates a string of roman val_10 for a given integer.
    e.g.
    >>> compute_1(89)
    'LXXXIX'
    >>> compute_1(4)
    'IV'
    """

    val_10 = ""

    val_7 = val_9 // 1000
    val_10 += val_7 * "M"
    val_9 %= 1000

    val_2 = val_9 // 100
    if val_2 == 9:
        val_10 += "CM"
        val_2 -= 9
    elif val_2 == 4:
        val_10 += "CD"
        val_2 -= 4
    if val_2 >= 5:
        val_10 += "D"
        val_2 -= 5
    val_10 += val_2 * "C"
    val_9 %= 100

    val_16 = val_9 // 10
    if val_16 == 9:
        val_10 += "XC"
        val_16 -= 9
    elif val_16 == 4:
        val_10 += "XL"
        val_16 -= 4
    if val_16 >= 5:
        val_10 += "L"
        val_16 -= 5
    val_10 += val_16 * "X"
    val_9 %= 10

    if val_9 == 9:
        val_10 += "IX"
        val_9 -= 9
    elif val_9 == 4:
        val_10 += "IV"
        val_9 -= 4
    if val_9 >= 5:
        val_10 += "V"
        val_9 -= 5
    val_10 += val_9 * "I"

    return val_10


def compute_3(val_12: str = "/p089_roman.txt") -> int:
    """
    Calculates and returns the answer to project euler problem 89.

    >>> compute_3("/numeralcleanup_test.txt")
    16
    """

    val_13 = 0

    with open(os.path.dirname(__file__) + val_12) as val_4:
        val_6 = val_4.readlines()

    for val_5 in val_6:
        val_11 = val_5.strip()
        val_9 = compute_2(val_11)
        val_14 = compute_1(val_9)
        val_13 += len(val_11) - len(val_14)

    return val_13


if __name__ == "__main__":
    print(f"{compute_3() = }")
