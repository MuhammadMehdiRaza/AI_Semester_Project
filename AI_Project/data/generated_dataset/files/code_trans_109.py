from timeit import timeit


def fn_3(v1: int) -> int:
    """
    Count the v1 of set bits in a 32 bit integer
    >>> fn_3(25)
    3
    >>> fn_3(37)
    3
    >>> fn_3(21)
    3
    >>> fn_3(58)
    4
    >>> fn_3(0)
    0
    >>> fn_3(256)
    1
    >>> fn_3(-1)
    Traceback (most recent call last):
        ...
    ValueError: the value of input must not be negative
    """
    if v1 < 0:
        raise ValueError("the value of input must not be negative")
    v2 = 0
    while v1:
        v1 &= v1 - 1
        v2 += 1
    return v2


def fn_4(v1: int) -> int:
    """
    Count the v1 of set bits in a 32 bit integer
    >>> fn_4(25)
    3
    >>> fn_4(37)
    3
    >>> fn_4(21)
    3
    >>> fn_4(58)
    4
    >>> fn_4(0)
    0
    >>> fn_4(256)
    1
    >>> fn_4(-1)
    Traceback (most recent call last):
        ...
    ValueError: the value of input must not be negative
    """
    if v1 < 0:
        raise ValueError("the value of input must not be negative")
    v2 = 0
    while v1:
        if v1 % 2 == 1:
            v2 += 1
        v1 >>= 1
    return v2


def fn_1() -> None:
    """
    Benchmark code for comparing 2 functions, with different length int values.
    Brian Kernighan's algorithm is consistently faster than using modulo_operator.
    """

    def fn_2(v1: int) -> None:
        v3 = "import __main__ as z"
        print(f"Benchmark when {v1 = }:")
        print(f"{fn_4(v1) = }")
        v4 = timeit(
            f"z.fn_4({v1})", v3=v3
        )
        print(f"timeit() runs in {v4} seconds")
        print(f"{fn_3(v1) = }")
        v4 = timeit(
            f"z.fn_3({v1})",
            v3=v3,
        )
        print(f"timeit() runs in {v4} seconds")

    for v1 in (25, 37, 58, 0):
        fn_2(v1)
        print()


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    fn_1()
