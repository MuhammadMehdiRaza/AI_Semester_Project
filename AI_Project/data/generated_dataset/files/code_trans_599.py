"""
In this problem, we want to rotate the val_2 elements by 90, 180, 270
(counterclockwise)
Discussion in stackoverflow:
https://stackoverflow.com/questions/42519/how-do-you-rotate-a-two-dimensional-array
"""

from __future__ import annotations


def fn_1(val_3: int = 4) -> list[list[int]]:
    """
    >>> fn_1()
    [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    >>> fn_1(1)
    [[1]]
    >>> fn_1(-2)
    [[1, 2], [3, 4]]
    >>> fn_1(3)
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    >>> fn_1() == fn_1(4)
    True
    """
    val_3 = abs(val_3) or 4
    return [[1 + val_4 + val_5 * val_3 for val_4 in range(val_3)] for val_5 in range(val_3)]



def fn_7(val_2: list[list[int]]) -> list[list[int]]:

    """

    >>> fn_7(fn_1())
    [[4, 8, 12, 16], [3, 7, 11, 15], [2, 6, 10, 14], [1, 5, 9, 13]]

    >>> fn_7(fn_1()) == fn_8(fn_3(fn_1()))
    True
    """

    return fn_4(fn_8(val_2))
    # OR.. fn_8(fn_3(val_2))


def fn_5(val_2: list[list[int]]) -> list[list[int]]:
    """
    >>> fn_5(fn_1())
    [[16, 15, 14, 13], [12, 11, 10, 9], [8, 7, 6, 5], [4, 3, 2, 1]]
    >>> fn_5(fn_1()) == fn_3(fn_4(fn_1()))
    True
    """

    return fn_4(fn_3(val_2))
    # OR.. fn_3(fn_4(val_2))


def fn_6(val_2: list[list[int]]) -> list[list[int]]:
    """
    >>> fn_6(fn_1())
    [[13, 9, 5, 1], [14, 10, 6, 2], [15, 11, 7, 3], [16, 12, 8, 4]]
    >>> fn_6(fn_1()) == fn_8(fn_4(fn_1()))
    True
    """

    return fn_3(fn_8(val_2))
    # OR.. fn_8(fn_4(val_2))


def fn_8(val_2: list[list[int]]) -> list[list[int]]:
    val_2[:] = [list(val_4) for val_4 in zip(*val_2)]
    return val_2



def fn_4(val_2: list[list[int]]) -> list[list[int]]:
    val_2[:] = val_2[::-1]
    return val_2


def fn_3(val_2: list[list[int]]) -> list[list[int]]:

    val_2[:] = [val_4[::-1] for val_4 in val_2]
    return val_2


def fn_2(val_2: list[list[int]]) -> None:
    for val_1 in val_2:
        print(*val_1)


if __name__ == "__main__":
    val_2 = fn_1()
    print("\norigin:\n")
    fn_2(val_2)
    print("\nrotate 90 counterclockwise:\n")
    fn_2(fn_7(val_2))

    val_2 = fn_1()
    print("\norigin:\n")
    fn_2(val_2)
    print("\nrotate 180:\n")
    fn_2(fn_5(val_2))

    val_2 = fn_1()
    print("\norigin:\n")
    fn_2(val_2)
    print("\nrotate 270 counterclockwise:\n")

    fn_2(fn_6(val_2))
