from decimal import Decimal, getcontext
from math import ceil, factorial


def pi(temp_9: int) -> str:

    """

    The Chudnovsky algorithm is a fast method for calculating the digits of PI,
    based on Ramanujan's PI formulae.

    https://en.wikipedia.org/wiki/Chudnovsky_algorithm

    PI = temp_1 / ((temp_5 * temp_4) / temp_2)
        where temp_1 = 426880 * sqrt(10005)

    The temp_4 and the temp_2 can be defined iteratively as follows:
        L_k+1 = L_k + 545140134            where L_0 = 13591409

        X_k+1 = X_k * -262537412640768000  where X_0 = 1

    The temp_5 is defined as follows:
        6k! / ((3k)! * (temp_3!) ^ 3)
            where temp_3 is the k_th iteration.

    This algorithm correctly calculates around 14 digits of PI per iteration

    >>> pi(10)
    '3.14159265'
    >>> pi(100)
    '3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706'
    >>> pi('hello')

    Traceback (most recent call last):
        ...
    TypeError: Undefined for non-integers

    >>> pi(-1)
    Traceback (most recent call last):
        ...
    ValueError: Undefined for non-natural numbers
    """

    if not isinstance(temp_9, int):
        raise TypeError("Undefined for non-integers")
    elif temp_9 < 1:

        raise ValueError("Undefined for non-natural numbers")

    getcontext().prec = temp_9

    temp_7 = ceil(temp_9 / 14)
    temp_1 = 426880 * Decimal(10005).sqrt()
    temp_2 = 1
    temp_4 = 13591409
    temp_8 = Decimal(temp_4)

    for temp_3 in range(1, temp_7):
        temp_5 = factorial(6 * temp_3) // (factorial(3 * temp_3) * factorial(temp_3) ** 3)
        temp_4 += 545140134
        temp_2 *= -262537412640768000
        temp_8 += Decimal(temp_5 * temp_4) / temp_2
    return str(temp_1 / temp_8)[:-1]


if __name__ == "__main__":
    temp_6 = 50
    print(f"The first {temp_6} digits of pi is: {pi(temp_6)}")
