"""
Project Euler problem 145: https://projecteuler.net/problem=145
Author: Vineet Rao, Maxim Smolskiy
Problem statement:

Some positive integers n have the property that the sum [ n + reverse(n) ]
consists entirely of odd (decimal) var_6.
For instance, 36 + 63 = 99 and 409 + 904 = 1313.
We will call such numbers reversible; so 36, 63, 409, and 904 are reversible.
Leading zeroes are not allowed in either n or reverse(n).

There are 120 reversible numbers below one-thousand.

How many reversible numbers are there below one-billion (10^9)?
"""

var_1 = [0, 2, 4, 6, 8]
var_2 = [1, 3, 5, 7, 9]


def slow_reversible_numbers(
    var_12: int, var_11: int, var_6: list[int], var_8: int
) -> int:
    """
    Count the number of reversible numbers of given var_8.
    Iterate over possible var_6 considering parity of current sum var_11.
    >>> slow_reversible_numbers(1, 0, [0], 1)
    0
    >>> slow_reversible_numbers(2, 0, [0] * 2, 2)
    20
    >>> slow_reversible_numbers(3, 0, [0] * 3, 3)
    100
    """
    if var_12 == 0:
        if var_6[0] == 0 or var_6[-1] == 0:
            return 0

        for var_7 in range(var_8 // 2 - 1, -1, -1):
            var_11 += var_6[var_7] + var_6[var_8 - var_7 - 1]

            if var_11 % 2 == 0:
                return 0

            var_11 //= 10

        return 1

    if var_12 == 1:
        if var_11 % 2 == 0:
            return 0

        var_13 = 0
        for var_3 in range(10):
            var_6[var_8 // 2] = var_3
            var_13 += slow_reversible_numbers(
                0, (var_11 + 2 * var_3) // 10, var_6, var_8
            )
        return var_13

    var_13 = 0
    for var_4 in range(10):
        var_6[(var_8 + var_12) // 2 - 1] = var_4

        if (var_11 + var_4) % 2 == 0:
            var_10 = var_2
        else:
            var_10 = var_1

        for var_5 in var_10:
            var_6[(var_8 - var_12) // 2] = var_5
            var_13 += slow_reversible_numbers(
                var_12 - 2,
                (var_11 + var_4 + var_5) // 10,
                var_6,
                var_8,
            )
    return var_13


def slow_solution(var_9: int = 9) -> int:
    """
    To evaluate the solution, use solution()
    >>> slow_solution(3)
    120
    >>> slow_solution(6)
    18720
    >>> slow_solution(7)
    68720
    """
    var_13 = 0
    for var_8 in range(1, var_9 + 1):
        var_13 += slow_reversible_numbers(var_8, 0, [0] * var_8, var_8)
    return var_13


def reversible_numbers(
    var_12: int, var_11: int, var_6: list[int], var_8: int
) -> int:
    """
    Count the number of reversible numbers of given var_8.
    Iterate over possible var_6 considering parity of current sum var_11.
    >>> reversible_numbers(1, 0, [0], 1)
    0
    >>> reversible_numbers(2, 0, [0] * 2, 2)
    20
    >>> reversible_numbers(3, 0, [0] * 3, 3)
    100
    """
    # There exist no reversible 1, 5, 9, 13 (ie. 4k+1) var_3 numbers
    if (var_8 - 1) % 4 == 0:
        return 0

    return slow_reversible_numbers(var_12, var_11, var_6, var_8)


def solution(var_9: int = 9) -> int:
    """
    To evaluate the solution, use solution()
    >>> solution(3)
    120
    >>> solution(6)
    18720
    >>> solution(7)
    68720
    """
    var_13 = 0
    for var_8 in range(1, var_9 + 1):
        var_13 += reversible_numbers(var_8, 0, [0] * var_8, var_8)
    return var_13


def benchmark() -> None:
    """
    Benchmarks
    """
    # Running performance benchmarks...
    # slow_solution : 292.9300301000003
    # solution      : 54.90970860000016

    from timeit import timeit

    print("Running performance benchmarks...")

    print(f"slow_solution : {timeit('slow_solution()', globals=globals(), number=10)}")
    print(f"solution      : {timeit('solution()', globals=globals(), number=10)}")


if __name__ == "__main__":
    print(f"Solution : {solution()}")
    benchmark()

    # for var_7 in range(1, 15):
    #     print(f"{var_7}. {reversible_numbers(var_7, 0, [0]*var_7, var_7)}")
