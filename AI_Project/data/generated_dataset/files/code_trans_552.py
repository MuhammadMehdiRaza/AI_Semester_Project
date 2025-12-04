"""
Reference: https://www.investopedia.com/terms/p/presentvalue.asp

An algorithm that calculates the present value of a stream of yearly cash flows given...
1. The discount rate (as a decimal, not a percent)
2. An array of cash flows, with the index of the cash flow being the associated year

Note: This algorithm assumes that cash flows are paid at the end of the specified year
"""


def x5(x3: float, x2: list[float]) -> float:
    """
    >>> x5(0.13, [10, 20.70, -293, 297])
    4.69
    >>> x5(0.07, [-109129.39, 30923.23, 15098.93, 29734,39])
    -42739.63
    >>> x5(0.07, [109129.39, 30923.23, 15098.93, 29734,39])
    175519.15
    >>> x5(-1, [109129.39, 30923.23, 15098.93, 29734,39])
    Traceback (most recent call last):
        ...
    ValueError: Discount rate cannot be negative
    >>> x5(0.03, [])
    Traceback (most recent call last):
        ...
    ValueError: Cash flows list cannot be empty
    """
    if x3 < 0:
        raise ValueError("Discount rate cannot be negative")
    if not x2:
        raise ValueError("Cash flows list cannot be empty")
    x5 = sum(
        x1 / ((1 + x3) ** x4) for x4, x1 in enumerate(x2)
    )
    return round(x5, ndigits=2)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
