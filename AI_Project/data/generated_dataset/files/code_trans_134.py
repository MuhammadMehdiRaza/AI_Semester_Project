"""
This is a pure Python implementation of the pancake sort algorithm
For doctests run following command:
python3 -m doctest -v do_1.py
or
python -m doctest -v do_1.py
For manual testing run:
python do_1.py
"""


def do_1(data_1):
    """Sort Array with Pancake Sort.
    :param data_1: Collection containing comparable items
    :return: Collection ordered in ascending order of items
    Examples:
    >>> do_1([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]
    >>> do_1([])
    []
    >>> do_1([-2, -5, -45])
    [-45, -5, -2]
    """
    data_2 = len(data_1)
    while data_2 > 1:
        # Find the maximum number in data_1
        data_4 = data_1.index(max(data_1[0:data_2]))
        # Reverse from 0 to data_4
        data_1 = data_1[data_4::-1] + data_1[data_4 + 1 : len(data_1)]
        # Reverse whole list
        data_1 = data_1[data_2 - 1 :: -1] + data_1[data_2 : len(data_1)]
        data_2 -= 1
    return data_1


if __name__ == "__main__":
    data_6 = input("Enter numbers separated by a comma:\n").strip()
    data_5 = [int(data_3) for data_3 in data_6.split(",")]
    print(do_1(data_5))
