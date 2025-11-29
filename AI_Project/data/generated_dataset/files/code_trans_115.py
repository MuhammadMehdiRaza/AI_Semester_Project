"""
Problem Description:
Given a binary tree, return its mirror.
"""



def fn_2(var_2: dict, var_6: int):
    if not var_6 or var_6 not in var_2:
        return
    var_3, var_5 = var_2[var_6][:2]
    var_2[var_6] = [var_5, var_3]
    fn_2(var_2, var_3)
    fn_2(var_2, var_5)


def fn_1(var_1: dict, var_6: int = 1) -> dict:

    """
    >>> fn_1({ 1: [2,3], 2: [4,5], 3: [6,7], 7: [8,9]}, 1)
    {1: [3, 2], 2: [5, 4], 3: [7, 6], 7: [9, 8]}
    >>> fn_1({ 1: [2,3], 2: [4,5], 3: [6,7], 4: [10,11]}, 1)
    {1: [3, 2], 2: [5, 4], 3: [7, 6], 4: [11, 10]}
    >>> fn_1({ 1: [2,3], 2: [4,5], 3: [6,7], 4: [10,11]}, 5)
    Traceback (most recent call last):
        ...
    ValueError: var_6 5 is not present in the var_1
    >>> fn_1({}, 5)
    Traceback (most recent call last):
        ...
    ValueError: binary tree cannot be empty

    """
    if not var_1:
        raise ValueError("binary tree cannot be empty")
    if var_6 not in var_1:
        var_4 = f"var_6 {var_6} is not present in the var_1"
        raise ValueError(var_4)
    var_2 = dict(var_1)
    fn_2(var_2, var_6)
    return var_2


if __name__ == "__main__":
    var_1 = {1: [2, 3], 2: [4, 5], 3: [6, 7], 7: [8, 9]}
    print(f"Binary tree: {var_1}")
    var_2 = fn_1(var_1, 5)
    print(f"Binary tree mirror: {var_2}")
