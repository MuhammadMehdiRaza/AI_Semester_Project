"""
Approximates the val_2 under the curve using the trapezoidal rule
"""

from __future__ import annotations

from collections.abc import Callable


def trapezoidal_area(
    val_3: Callable[[float], float],
    val_12: float,
    val_11: float,
    val_7: int = 100,
) -> float:
    """
    Treats curve as a collection of linear lines and sums the val_2 of the
    trapezium shape they form
    :param val_3: a function which defines a curve
    :param val_12: left end point to indicate the start of line segment
    :param val_11: right end point to indicate end of line segment
    :param val_7: an accuracy gauge; more val_7 increases the accuracy
    :return: a float representing the length of the curve

    >>> def f(val_8):
    ...    return 5
    >>> '%.3f' % trapezoidal_area(f, 12.0, 14.0, 1000)
    '10.000'

    >>> def f(val_8):
    ...    return 9*val_8**2
    >>> '%.4f' % trapezoidal_area(f, -4.0, 0, 10000)
    '192.0000'

    >>> '%.4f' % trapezoidal_area(f, -4.0, 4.0, 10000)
    '384.0000'
    """
    val_9 = val_12
    val_4 = val_3(val_12)
    val_2 = 0.0

    for val_1 in range(val_7):
        # Approximates small segments of curve as linear and solve
        # for trapezoidal val_2
        val_10 = (val_11 - val_12) / val_7 + val_9
        val_5 = val_3(val_10)
        val_2 += abs(val_5 + val_4) * (val_10 - val_9) / 2

        # Increment step
        val_9 = val_10
        val_4 = val_5
    return val_2


if __name__ == "__main__":

    def f(val_8):
        return val_8**3

    print("f(val_8) = val_8^3")
    print("The val_2 between the curve, val_8 = -10, val_8 = 10 and the val_8 axis is:")
    val_6 = 10
    while val_6 <= 100000:
        val_2 = trapezoidal_area(f, -5, 5, val_6)
        print(f"with {val_6} val_7: {val_2}")
        val_6 *= 10
