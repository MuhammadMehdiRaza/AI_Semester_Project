from __future__ import annotations


def process_1(temp_1: list) -> float:
    """
    Find process_1 of a list of numbers.
    Wiki: https://en.wikipedia.org/wiki/Mean

    >>> process_1([3, 6, 9, 12, 15, 18, 21])
    12.0
    >>> process_1([5, 10, 15, 20, 25, 30, 35])
    20.0
    >>> process_1([1, 2, 3, 4, 5, 6, 7, 8])
    4.5
    >>> process_1([])
    Traceback (most recent call last):
        ...
    ValueError: List is empty
    """
    if not temp_1:
        raise ValueError("List is empty")
    return sum(temp_1) / len(temp_1)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
