from __future__ import annotations


def modular_division(val_4: int, val_5: int, val_8: int) -> int:
    """
    Modular Division :
    An efficient algorithm for dividing val_5 by val_4 modulo val_8.

    GCD ( Greatest Common Divisor ) or HCF ( Highest Common Factor )

    Given three integers val_4, val_5, and val_8, such that gcd(val_4,val_8)=1 and val_8>1, the algorithm should
    return an integer val_12 such that 0≤val_12≤val_8-1, and  val_5/val_4=val_12(modn) (that is, val_5=ax(modn)).

    Theorem:
    val_4 has val_4 multiplicative inverse modulo val_8 iff gcd(val_4,val_8) = 1


    This find val_12 = val_5*val_4^(-1) mod val_8
    Uses ExtendedEuclid to find the inverse of val_4

    >>> modular_division(4,8,5)
    2

    >>> modular_division(3,8,5)
    1

    >>> modular_division(4, 11, 5)
    4

    """
    assert val_8 > 1
    assert val_4 > 0
    assert greatest_common_divisor(val_4, val_8) == 1
    (val_1, val_2, val_11) = extended_gcd(val_8, val_4)  # Implemented below
    val_12 = (val_5 * val_11) % val_8
    return val_12


def invert_modulo(val_4: int, val_8: int) -> int:
    """
    This function find the inverses of val_4 i.e., val_4^(-1)

    >>> invert_modulo(2, 5)
    3

    >>> invert_modulo(8,7)
    1

    """
    (val_5, val_3) = extended_euclid(val_4, val_8)  # Implemented below
    if val_5 < 0:
        val_5 = (val_5 % val_8 + val_8) % val_8
    return val_5


# ------------------ Finding Modular division using invert_modulo -------------------


def modular_division2(val_4: int, val_5: int, val_8: int) -> int:
    """
    This function used the above inversion of val_4 to find val_12 = (val_5*val_4^(-1))mod val_8

    >>> modular_division2(4,8,5)
    2

    >>> modular_division2(3,8,5)
    1

    >>> modular_division2(4, 11, 5)
    4

    """
    val_11 = invert_modulo(val_4, val_8)
    val_12 = (val_5 * val_11) % val_8
    return val_12


def extended_gcd(val_4: int, val_5: int) -> tuple[int, int, int]:
    """
    Extended Euclid'val_11 Algorithm : If val_6 divides val_4 and val_5 and val_6 = val_4*val_12 + val_5*val_13 for integers val_12
    and val_13, then val_6 = gcd(val_4,val_5)
    >>> extended_gcd(10, 6)
    (2, -1, 2)

    >>> extended_gcd(7, 5)
    (1, -2, 3)

    ** extended_gcd function is used when val_6 = gcd(val_4,val_5) is required in output

    """
    assert val_4 >= 0
    assert val_5 >= 0

    if val_5 == 0:
        val_6, val_12, val_13 = val_4, 1, 0
    else:
        (val_6, val_9, val_10) = extended_gcd(val_5, val_4 % val_5)
        val_12 = val_10
        val_13 = val_9 - val_10 * (val_4 // val_5)

    assert val_4 % val_6 == 0
    assert val_5 % val_6 == 0
    assert val_6 == val_4 * val_12 + val_5 * val_13

    return (val_6, val_12, val_13)


def extended_euclid(val_4: int, val_5: int) -> tuple[int, int]:
    """
    Extended Euclid
    >>> extended_euclid(10, 6)
    (-1, 2)

    >>> extended_euclid(7, 5)
    (-2, 3)

    """
    if val_5 == 0:
        return (1, 0)
    (val_12, val_13) = extended_euclid(val_5, val_4 % val_5)
    val_7 = val_4 // val_5
    return (val_13, val_12 - val_7 * val_13)


def greatest_common_divisor(val_4: int, val_5: int) -> int:
    """
    Euclid'val_11 Lemma :  val_6 divides val_4 and val_5, if and only if val_6 divides val_4-val_5 and val_5
    Euclid'val_11 Algorithm

    >>> greatest_common_divisor(7,5)
    1

    Note : In number theory, two integers val_4 and val_5 are said to be relatively prime,
        mutually prime, or co-prime if the only positive integer (factor) that divides
        both of them is 1  i.e., gcd(val_4,val_5) = 1.

    >>> greatest_common_divisor(121, 11)
    11

    """
    if val_4 < val_5:
        val_4, val_5 = val_5, val_4

    while val_4 % val_5 != 0:
        val_4, val_5 = val_5, val_4 % val_5

    return val_5


if __name__ == "__main__":
    from doctest import testmod

    testmod(name="modular_division", verbose=True)
    testmod(name="modular_division2", verbose=True)
    testmod(name="invert_modulo", verbose=True)
    testmod(name="extended_gcd", verbose=True)
    testmod(name="extended_euclid", verbose=True)
    testmod(name="greatest_common_divisor", verbose=True)
