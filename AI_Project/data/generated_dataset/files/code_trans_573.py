def compute_1(v2: str) -> int:
    """
    Count the number of v3 in a given string.

    :param v2: Input string to count v3 in.
    :return: Number of v3 in the input string.

    Examples:
    >>> compute_1("hello world")
    3
    >>> compute_1("HELLO WORLD")
    3
    >>> compute_1("123 hello world")
    3
    >>> compute_1("")
    0
    >>> compute_1("a quick brown fox")
    5
    >>> compute_1("the quick BROWN fox")
    5
    >>> compute_1("PYTHON")
    1
    """
    if not isinstance(v2, str):
        raise ValueError("Input must be a string")

    v3 = "aeiouAEIOU"
    return sum(1 for v1 in v2 if v1 in v3)


if __name__ == "__main__":
    from doctest import testmod

    testmod()
