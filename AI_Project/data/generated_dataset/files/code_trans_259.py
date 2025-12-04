"""
Program to check if a cycle is present in a given data_1
"""


def check_cycle(data_1: dict) -> bool:
    """
    Returns True if data_1 is cyclic else False
    >>> check_cycle(data_1={0:[], 1:[0, 3], 2:[0, 4], 3:[5], 4:[5], 5:[]})
    False
    >>> check_cycle(data_1={0:[1, 2], 1:[2], 2:[0, 3], 3:[3]})
    True
    """
    # Keep track of data_5 nodes
    data_5: set[int] = set()
    # To detect a back edge, keep track of vertices currently in the recursion stack
    data_3: set[int] = set()
    return any(
        data_2 not in data_5 and depth_first_search(data_1, data_2, data_5, data_3)
        for data_2 in data_1
    )


def depth_first_search(data_1: dict, data_4: int, data_5: set, data_3: set) -> bool:
    """
    Recur for all neighbours.
    If any neighbour is data_5 and in data_3 then data_1 is cyclic.
    >>> data_1 = {0:[], 1:[0, 3], 2:[0, 4], 3:[5], 4:[5], 5:[]}
    >>> data_4, data_5, data_3 = 0, set(), set()
    >>> depth_first_search(data_1, data_4, data_5, data_3)
    False
    """
    # Mark current data_2 as data_5 and add to recursion stack
    data_5.add(data_4)
    data_3.add(data_4)

    for data_2 in data_1[data_4]:
        if data_2 not in data_5:
            if depth_first_search(data_1, data_2, data_5, data_3):
                return True
        elif data_2 in data_3:
            return True

    # The data_2 needs to be removed from recursion stack before function ends
    data_3.remove(data_4)
    return False


if __name__ == "__main__":
    from doctest import testmod

    testmod()
