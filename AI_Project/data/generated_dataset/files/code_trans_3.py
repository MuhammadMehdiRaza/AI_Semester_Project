"""

Task:
Given a positive int val_2. Return True if this val_2 is power of 4
or False otherwise.

Implementation notes: Use bit manipulation.
For example if the val_2 is the power of 2 it's bits representation:
n     = 0..100..00
n - 1 = 0..011..11

n & (n - 1) - no intersections = 0
If the val_2 is a power of 4 then it should be a power of 2
and the set bit should be at an odd position.
"""


def power_of_4(val_2: int) -> bool:
    """
    Return True if this val_2 is power of 4 or False otherwise.

    >>> power_of_4(0)
    Traceback (most recent call last):
        ...
    ValueError: val_2 must be positive
    >>> power_of_4(1)
    True
    >>> power_of_4(2)
    False
    >>> power_of_4(4)
    True
    >>> power_of_4(6)
    False
    >>> power_of_4(8)
    False
    >>> power_of_4(17)
    False
    >>> power_of_4(64)
    True
    >>> power_of_4(-1)
    Traceback (most recent call last):
        ...
    ValueError: val_2 must be positive
    >>> power_of_4(1.2)
    Traceback (most recent call last):
        ...
    TypeError: val_2 must be an integer

    """
    if not isinstance(val_2, int):
        raise TypeError("val_2 must be an integer")
    if val_2 <= 0:
        raise ValueError("val_2 must be positive")
    if val_2 & (val_2 - 1) == 0:
        val_1 = 0
        while val_2:
            val_1 += 1
            val_2 >>= 1
        return val_1 % 2 == 1
    else:
        return False


if __name__ == "__main__":
    import doctest

    doctest.testmod()
