def compute_1(x2: int, x1: int) -> str:
    """
    >>> print(compute_1(0, 0xFFFF))
        0: 00000000
    65535: 1111111111111111
    """
    return f"{x2:>5}: {x2:08b}\n{x1:>5}: {x1:08b}"


def compute_2(x5: int) -> int:
    """
    1. We use bitwise AND operations to separate the even bits (0, 2, 4, 6, etc.) and
       odd bits (1, 3, 5, 7, etc.) in the input number.
    2. We then right-shift the even bits by 1 position and left-shift the odd bits by
       1 position to swap them.
    3. Finally, we combine the swapped even and odd bits using a bitwise OR operation
       to obtain the final result.
    >>> print(compute_1(0, compute_2(0)))
        0: 00000000
        0: 00000000
    >>> print(compute_1(1, compute_2(1)))
        1: 00000001
        2: 00000010
    >>> print(compute_1(2, compute_2(2)))
        2: 00000010
        1: 00000001
    >>> print(compute_1(3, compute_2(3)))
        3: 00000011
        3: 00000011
    >>> print(compute_1(4, compute_2(4)))
        4: 00000100
        8: 00001000
    >>> print(compute_1(5, compute_2(5)))
        5: 00000101
       10: 00001010
    >>> print(compute_1(6, compute_2(6)))
        6: 00000110
        9: 00001001
    >>> print(compute_1(23, compute_2(23)))
       23: 00010111
       43: 00101011
    """
    # Get all even bits - 0xAAAAAAAA is a 32-bit number with all even bits set to 1
    x3 = x5 & 0xAAAAAAAA

    # Get all odd bits - 0x55555555 is a 32-bit number with all odd bits set to 1
    x6 = x5 & 0x55555555

    # Right shift even bits and left shift odd bits and swap them
    return x3 >> 1 | x6 << 1


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    for x4 in (-1, 0, 1, 2, 3, 4, 23, 24):
        print(compute_1(x4, compute_2(x4)), "\n")
