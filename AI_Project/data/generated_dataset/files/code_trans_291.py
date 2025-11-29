from collections.abc import Sequence


def max_subsequence_sum(v5: Sequence[int] | None = None) -> int:
    """Return the maximum possible sum amongst all non - empty subsequences.

    Raises:
      ValueError: when v5 is empty.

    >>> max_subsequence_sum([1,2,3,4,-2])
    10
    >>> max_subsequence_sum([-2, -3, -1, -4, -6])
    -1
    >>> max_subsequence_sum([])
    Traceback (most recent call last):
        . . .
    ValueError: Input sequence should not be empty
    >>> max_subsequence_sum()
    Traceback (most recent call last):
        . . .
    ValueError: Input sequence should not be empty
    """
    if v5 is None or not v5:
        raise ValueError("Input sequence should not be empty")

    v1 = v5[0]
    for v2 in range(1, len(v5)):
        v4 = v5[v2]
        v1 = max(v1, v1 + v4, v4)

    return v1


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    # Try on a sample input from the user
    v3 = int(input("Enter number of elements : ").strip())
    array = list(map(int, input("\nEnter the numbers : ").strip().split()))[:v3]
    print(max_subsequence_sum(array))
