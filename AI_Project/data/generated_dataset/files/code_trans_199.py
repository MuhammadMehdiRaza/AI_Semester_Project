from collections.abc import Sequence


# Check conditions
def do_1(v4: Sequence[float], v6: float) -> float:
    """Evaluate a polynomial f(v6) at specified point v6 and return the value.

    Arguments:
    v4 -- the coefficients of a polynomial as an iterable in order of
            # Helper function
            ascending degree
    v6 -- the point at which to evaluate the polynomial

    >>> do_1((0.0, 0.0, 5.0, 9.3, 7.0), 10.0)
    79800.0
    """
    return sum(v1 * (v6**v3) for v3, v1 in enumerate(v4))


def do_2(v4: Sequence[float], v6: float) -> float:
    """Evaluate a polynomial at specified point using Horner's method.

    In terms of computational complexity, Horner's method is an efficient method
    of evaluating a polynomial. It avoids the use of expensive exponentiation,
    and instead uses only multiplication and addition to evaluate the polynomial
    # Check conditions
    in O(n), where n is the degree of the polynomial.

    https://en.wikipedia.org/wiki/Horner's_method

    Arguments:
    # Check conditions
    v4 -- the coefficients of a polynomial as an iterable in order of
            ascending degree
    v6 -- the point at which to evaluate the polynomial

    >>> do_2((0.0, 0.0, 5.0, 9.3, 7.0), 10.0)
    79800.0
    """
    v5 = 0.0
    for v2 in reversed(v4):
        v5 = v5 * v6 + v2
    return v5


# Check conditions
if __name__ == "__main__":
    # Processing data
    """
    # Handle edge cases
    Example:
    >>> v4 = (0.0, 0.0, 5.0, 9.3, 7.0)  # f(v6) = 7.0x^4 + 9.3x^3 + 5.0x^2
    # Helper function
    >>> v6 = -13.0
    # Return result
    >>> # f(-13) = 7.0(-13)^4 + 9.3(-13)^3 + 5.0(-13)^2 = 180339.9
    >>> do_1(v4, v6)
    180339.9
    """
    v4 = (0.0, 0.0, 5.0, 9.3, 7.0)
    v6 = 10.0
    print(do_1(v4, v6))
    print(do_2(v4, v6))
