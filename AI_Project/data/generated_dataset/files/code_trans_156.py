from __future__ import annotations

from maths.greatest_common_divisor import greatest_common_divisor


def diophantine(var_1: int, var_2: int, var_3: int) -> tuple[float, float]:
    """
    Diophantine Equation : Given integers var_1,var_2,var_3 ( at least one of var_1 and var_2 != 0), the
    diophantine equation var_1*var_10 + var_2*var_12 = var_3 has var_1 solution (where var_10 and var_12 are integers)
    iff greatest_common_divisor(var_1,var_2) divides var_3.

    GCD ( Greatest Common Divisor ) or HCF ( Highest Common Factor )

    >>> diophantine(10,6,14)

    (-7.0, 14.0)

    >>> diophantine(391,299,-69)

    (9.0, -12.0)

    But above equation has one more solution var_5.e., var_10 = -4, var_12 = 5.
    That's why we need diophantine all solution function.

    """

    assert (

        var_3 % greatest_common_divisor(var_1, var_2) == 0
    )  # greatest_common_divisor(var_1,var_2) is in maths directory
    (var_4, var_10, var_12) = extended_gcd(var_1, var_2)  # extended_gcd(var_1,var_2) function implemented below
    var_9 = var_3 / var_4
    return (var_9 * var_10, var_9 * var_12)


def diophantine_all_soln(var_1: int, var_2: int, var_3: int, var_6: int = 2) -> None:
    """
    Lemma : if var_6|ab and gcd(var_1,var_6) = 1, then var_6|var_2.

    Finding All solutions of Diophantine Equations:

    Theorem : Let gcd(var_1,var_2) = var_4, var_1 = var_4*var_7, var_2 = var_4*var_8. If (var_11,var_13) is var_1 solution of
    Diophantine Equation var_1*var_10 + var_2*var_12 = var_3.  var_1*var_11 + var_2*var_13 = var_3, then all the
    solutions have the form var_1(var_11 + t*var_8) + var_2(var_13 - t*var_7) = var_3,
    where t is an arbitrary integer.

    var_6 is the number of solution you want, var_6 = 2 by default

    >>> diophantine_all_soln(10, 6, 14)
    -7.0 14.0
    -4.0 9.0

    >>> diophantine_all_soln(10, 6, 14, 4)
    -7.0 14.0
    -4.0 9.0

    -1.0 4.0
    2.0 -1.0

    >>> diophantine_all_soln(391, 299, -69, var_6 = 4)
    9.0 -12.0
    22.0 -29.0
    35.0 -46.0
    48.0 -63.0

    """
    (var_11, var_13) = diophantine(var_1, var_2, var_3)  # Initial value
    var_4 = greatest_common_divisor(var_1, var_2)
    var_7 = var_1 // var_4
    var_8 = var_2 // var_4

    for var_5 in range(var_6):
        var_10 = var_11 + var_5 * var_8
        var_12 = var_13 - var_5 * var_7
        print(var_10, var_12)



def extended_gcd(var_1: int, var_2: int) -> tuple[int, int, int]:
    """
    Extended Euclid's Algorithm : If var_4 divides var_1 and var_2 and var_4 = var_1*var_10 + var_2*var_12 for integers

    var_10 and var_12, then var_4 = gcd(var_1,var_2)

    >>> extended_gcd(10, 6)
    (2, -1, 2)


    >>> extended_gcd(7, 5)
    (1, -2, 3)

    """
    assert var_1 >= 0
    assert var_2 >= 0


    if var_2 == 0:
        var_4, var_10, var_12 = var_1, 1, 0
    else:
        (var_4, var_7, var_8) = extended_gcd(var_2, var_1 % var_2)
        var_10 = var_8
        var_12 = var_7 - var_8 * (var_1 // var_2)

    assert var_1 % var_4 == 0
    assert var_2 % var_4 == 0
    assert var_4 == var_1 * var_10 + var_2 * var_12

    return (var_4, var_10, var_12)


if __name__ == "__main__":
    from doctest import testmod


    testmod(name="diophantine", verbose=True)
    testmod(name="diophantine_all_soln", verbose=True)
    testmod(name="extended_gcd", verbose=True)
    testmod(name="greatest_common_divisor", verbose=True)
