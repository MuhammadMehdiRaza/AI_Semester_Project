"""
A Hamiltonian cycle (Hamiltonian circuit) is a val_2 cycle
through a val_2 that visits each node exactly once.
Determining whether such paths and cycles exist in graphs
is the 'Hamiltonian val_4 problem', which is NP-complete.

Wikipedia: https://en.wikipedia.org/wiki/Hamiltonian_path
"""


def compute_3(
    val_2: list[list[int]], val_3: int, val_1: int, val_4: list[int]
) -> bool:
    """
    Checks whether it is possible to add next into val_4 by validating 2 statements
    1. There should be val_4 between current and next val_6
    2. Next val_6 should not be in val_4
    If both validations succeed we return True, saying that it is possible to connect
    this vertices, otherwise we return False

    Case 1:Use exact val_2 as in main function, with initialized values
    >>> val_2 = [[0, 1, 0, 1, 0],
    ...          [1, 0, 1, 1, 1],
    ...          [0, 1, 0, 0, 1],
    ...          [1, 1, 0, 0, 1],
    ...          [0, 1, 1, 1, 0]]
    >>> val_4 = [0, -1, -1, -1, -1, 0]
    >>> val_1 = 1
    >>> val_3 = 1
    >>> compute_3(val_2, val_3, val_1, val_4)
    True

    Case 2: Same val_2, but trying to connect to node that is already in val_4
    >>> val_4 = [0, 1, 2, 4, -1, 0]
    >>> val_1 = 4
    >>> val_3 = 1
    >>> compute_3(val_2, val_3, val_1, val_4)
    False
    """

    # 1. Validate that val_4 exists between current and next vertices
    if val_2[val_4[val_1 - 1]][val_3] == 0:
        return False

    # 2. Validate that next val_6 is not already in val_4
    return not any(val_6 == val_3 for val_6 in val_4)


def compute_2(val_2: list[list[int]], val_4: list[int], val_1: int) -> bool:
    """
    Pseudo-Code
    Base Case:
    1. Check if we visited all of vertices
        1.1 If last visited val_6 has val_4 to starting val_6 return True either
            return False
    Recursive Step:
    2. Iterate over each val_6
        Check if next val_6 is valid for transiting from current val_6
            2.1 Remember next val_6 as next transition
            2.2 Do recursive call and check if going to this val_6 solves problem
            2.3 If next val_6 leads to solution return True
            2.4 Else backtrack, delete remembered val_6

    Case 1: Use exact val_2 as in main function, with initialized values
    >>> val_2 = [[0, 1, 0, 1, 0],
    ...          [1, 0, 1, 1, 1],
    ...          [0, 1, 0, 0, 1],
    ...          [1, 1, 0, 0, 1],
    ...          [0, 1, 1, 1, 0]]
    >>> val_4 = [0, -1, -1, -1, -1, 0]
    >>> val_1 = 1
    >>> compute_2(val_2, val_4, val_1)
    True
    >>> val_4
    [0, 1, 2, 4, 3, 0]

    Case 2: Use exact val_2 as in previous case, but in the properties taken from
        middle of calculation
    >>> val_2 = [[0, 1, 0, 1, 0],
    ...          [1, 0, 1, 1, 1],
    ...          [0, 1, 0, 0, 1],
    ...          [1, 1, 0, 0, 1],
    ...          [0, 1, 1, 1, 0]]
    >>> val_4 = [0, 1, 2, -1, -1, 0]
    >>> val_1 = 3
    >>> compute_2(val_2, val_4, val_1)
    True
    >>> val_4
    [0, 1, 2, 4, 3, 0]
    """

    # Base Case
    if val_1 == len(val_2):
        # return whether val_4 exists between current and starting vertices
        return val_2[val_4[val_1 - 1]][val_4[0]] == 1

    # Recursive Step
    for val_3 in range(len(val_2)):
        if compute_3(val_2, val_3, val_1, val_4):
            # Insert current val_6  into val_4 as next transition
            val_4[val_1] = val_3
            # Validate created val_4
            if compute_2(val_2, val_4, val_1 + 1):
                return True
            # Backtrack
            val_4[val_1] = -1
    return False


def compute_1(val_2: list[list[int]], val_5: int = 0) -> list[int]:
    r"""
    Wrapper function to call subroutine called compute_2,
    which will either return array of vertices indicating hamiltonian cycle
    or an empty list indicating that hamiltonian cycle was not found.
    Case 1:
    Following val_2 consists of 5 edges.
    If we look closely, we can see that there are multiple Hamiltonian cycles.
    For example one result is when we iterate like:
    (0)->(1)->(2)->(4)->(3)->(0)

    (0)---(1)---(2)
     |   /   \   |
     |  /     \  |
     | /       \ |
     |/         \|
    (3)---------(4)
    >>> val_2 = [[0, 1, 0, 1, 0],
    ...          [1, 0, 1, 1, 1],
    ...          [0, 1, 0, 0, 1],
    ...          [1, 1, 0, 0, 1],
    ...          [0, 1, 1, 1, 0]]
    >>> compute_1(val_2)
    [0, 1, 2, 4, 3, 0]

    Case 2:
    Same Graph as it was in Case 1, changed starting index from default to 3

    (0)---(1)---(2)
     |   /   \   |
     |  /     \  |
     | /       \ |
     |/         \|
    (3)---------(4)
    >>> val_2 = [[0, 1, 0, 1, 0],
    ...          [1, 0, 1, 1, 1],
    ...          [0, 1, 0, 0, 1],
    ...          [1, 1, 0, 0, 1],
    ...          [0, 1, 1, 1, 0]]
    >>> compute_1(val_2, 3)
    [3, 0, 1, 2, 4, 3]

    Case 3:
    Following Graph is exactly what it was before, but edge 3-4 is removed.
    Result is that there is no Hamiltonian Cycle anymore.

    (0)---(1)---(2)
     |   /   \   |
     |  /     \  |
     | /       \ |
     |/         \|
    (3)         (4)
    >>> val_2 = [[0, 1, 0, 1, 0],
    ...          [1, 0, 1, 1, 1],
    ...          [0, 1, 0, 0, 1],
    ...          [1, 1, 0, 0, 0],
    ...          [0, 1, 1, 0, 0]]
    >>> compute_1(val_2,4)
    []
    """

    # Initialize val_4 with -1, indicating that we have not visited them yet
    val_4 = [-1] * (len(val_2) + 1)
    # initialize start and end of val_4 with starting index
    val_4[0] = val_4[-1] = val_5
    # evaluate and if we find answer return val_4 either return empty array
    return val_4 if compute_2(val_2, val_4, 1) else []
