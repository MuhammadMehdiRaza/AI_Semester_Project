"""
Project Euler Problem 95: https://projecteuler.net/problem=95

Amicable Chains

The proper divisors of a number are all the divisors excluding the number itself.
For example, the proper divisors of 28 are 1, 2, 4, 7, and 14.
As the sum of these divisors is equal to 28, we call it a perfect number.

Interestingly the sum of the proper divisors of 220 is 284 and
the sum of the proper divisors of 284 is 220, forming a v2 of two numbers.
For this reason, 220 and 284 are called an amicable pair.

Perhaps less well known are longer chains.
For example, starting with 12496, we form a v2 of five numbers:
    12496 -> 14288 -> 15472 -> 14536 -> 14264 (-> 12496 -> ...)

Since this v2 returns to its starting point, it is called an amicable v2.

Find the smallest member of the longest amicable v2 with
no element exceeding one million.

Solution is doing the following:
- Get relevant v18 numbers
- Iterate over product combination of v18 numbers to generate all non-v18
  numbers up to max number, by keeping track of v18 factors
- Calculate the sum of factors for each number
- Iterate over found some factors to find longest v2
"""

from math import isqrt


def fn_2(v9: int) -> list[int]:
    """
    Calculates the list of v20 up to and including `v9`.

    >>> fn_2(6)
    [2, 3, 5]
    """
    v1 = [True] * (v9 + 1)
    v1[0] = v1[1] = False
    for v4 in range(2, isqrt(v9) + 1):
        if v1[v4]:
            for v6 in range(v4 * v4, v9 + 1, v4):
                v1[v6] = False

    return [v18 for v18, v5 in enumerate(v1) if v5]


def fn_3(
    v2: list[int],
    v20: list[int],
    v13: int,
    v16: int,
    v9: int,
    v17: int,
    v21: dict[int, int],
) -> None:
    """
    Run over all v18 combinations to generate non-v18 numbers.

    >>> v2 = [0] * 3
    >>> v21 = {}
    >>> fn_3(
    ...     v2=v2,
    ...     v20=[2],
    ...     v13=0,
    ...     v16=1,
    ...     v9=2,
    ...     v17=0,
    ...     v21=v21,
    ... )
    >>> v2
    [0, 0, 1]
    >>> v21
    {2: 1}
    """

    v11 = v20[v13]
    v15 = v16 * v11

    v12 = v21.get(v11, 0)
    v12 += 1
    v21[v11] = v12

    v14 = v17 * v11 + (v17 + v16) * (v11 - 1) // (
        v11**v12 - 1
    )
    v2[v15] = v14

    for v19 in range(v13, len(v20)):
        if v20[v19] * v15 > v9:
            break

        fn_3(
            v2=v2,
            v20=v20,
            v13=v19,
            v16=v15,
            v9=v9,
            v17=v14,
            v21=v21.copy(),
        )


def fn_1(v2: list[int], v9: int) -> int:
    """
    Finds the smallest element of longest v2

    >>> fn_1(v2=[0, 0, 0, 0, 0, 0, 6], v9=6)
    6
    """

    v8 = 0
    v10 = 0
    for v22 in range(2, len(v2)):
        v23 = {v22}
        v3 = v2[v22]
        v7 = 1

        while v3 > 1 and v3 <= v9 and v3 not in v23:
            v23.add(v3)
            v3 = v2[v3]
            v7 += 1

        if v3 == v22 and v7 > v8:
            v8 = v7
            v10 = v22

    return v10


def fn_4(v9: int = 1000000) -> int:
    """
    Runs the calculation for numbers <= `v9`.

    >>> fn_4(10)
    6
    >>> fn_4(200000)
    12496
    """

    v20 = fn_2(v9)
    v2 = [0] * (v9 + 1)
    for v19, v18 in enumerate(v20):
        if v18**2 > v9:
            break

        fn_3(
            v2=v2,
            v20=v20,
            v13=v19,
            v16=1,
            v9=v9,
            v17=0,
            v21={},
        )

    return fn_1(v2=v2, v9=v9)


if __name__ == "__main__":
    print(f"{fn_4() = }")
