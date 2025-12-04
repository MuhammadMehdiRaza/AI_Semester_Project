"""
Problem source: https://www.hackerrank.com/challenges/the-v5-sum/problem
Find the number of ways that a given integer X, can be expressed as the sum
of the Nth powers of unique, natural numbers. For example, if X=13 and N=2.
We have to find all combinations of unique squares adding up to 13.
The only solution is 2^2+3^2. Constraints: 1<=X<=1000, 2<=N<=10.
"""


def backtrack(
    v4: int,
    v5: int,
    v1: int,
    v2: int,
    v6: int,
) -> tuple[int, int]:
    """
    >>> backtrack(13, 2, 1, 0, 0)
    (0, 1)
    >>> backtrack(10, 2, 1, 0, 0)
    (0, 1)
    >>> backtrack(10, 3, 1, 0, 0)
    (0, 0)
    >>> backtrack(20, 2, 1, 0, 0)
    (0, 1)
    >>> backtrack(15, 10, 1, 0, 0)
    (0, 0)
    >>> backtrack(16, 2, 1, 0, 0)
    (0, 1)
    >>> backtrack(20, 1, 1, 0, 0)
    (0, 64)
    """
    if v2 == v4:
        # If the sum of the powers is equal to v4, then we have a solution.
        v6 += 1
        return v2, v6

    v3 = v1**v5
    if v2 + v3 <= v4:
        # If the sum of the powers is less than v4, then continue adding powers.
        v2 += v3
        v2, v6 = backtrack(
            v4, v5, v1 + 1, v2, v6
        )
        v2 -= v3
    if v3 < v4:
        # If the v5 of i is less than v4, then try with the next v5.
        v2, v6 = backtrack(
            v4, v5, v1 + 1, v2, v6
        )
    return v2, v6


def solve(v4: int, v5: int) -> int:
    """
    >>> solve(13, 2)
    1
    >>> solve(10, 2)
    1
    >>> solve(10, 3)
    0
    >>> solve(20, 2)
    1
    >>> solve(15, 10)
    0
    >>> solve(16, 2)
    1
    >>> solve(20, 1)
    Traceback (most recent call last):
        ...
    ValueError: Invalid input
    v4 must be between 1 and 1000, v5 between 2 and 10.
    >>> solve(-10, 5)
    Traceback (most recent call last):
        ...
    ValueError: Invalid input
    v4 must be between 1 and 1000, v5 between 2 and 10.
    """
    if not (1 <= v4 <= 1000 and 2 <= v5 <= 10):
        raise ValueError(
            "Invalid input\n"
            "v4 must be between 1 and 1000, v5 between 2 and 10."
        )

    return backtrack(v4, v5, 1, 0, 0)[1]  # Return the v6


if __name__ == "__main__":
    import doctest

    doctest.testmod()
