"""
Python implementation of a sort algorithm.
Best Case Scenario : O(n)
Worst Case Scenario : O(n^2) because native Python functions:min, max and remove are
already O(n)
"""


def func_1(temp_1):
    """Pure implementation of the fastest merge sort algorithm in Python

    :param temp_1: some mutable ordered temp_1 with heterogeneous
    comparable items inside
    :return: a temp_1 ordered by ascending

    Examples:
    >>> func_1([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]

    >>> func_1([])
    []

    >>> func_1([-2, -5, -45])
    [-45, -5, -2]
    """
    temp_6, temp_2 = [], []
    while len(temp_1) > 1:
        temp_5, temp_4 = min(temp_1), max(temp_1)
        temp_6.append(temp_5)
        temp_2.append(temp_4)
        temp_1.remove(temp_5)
        temp_1.remove(temp_4)
    temp_2.reverse()
    return temp_6 + temp_1 + temp_2


if __name__ == "__main__":
    temp_8 = input("Enter numbers separated by a comma:\n").strip()
    temp_7 = [int(temp_3) for temp_3 in temp_8.split(",")]
    print(*func_1(temp_7), sep=",")
