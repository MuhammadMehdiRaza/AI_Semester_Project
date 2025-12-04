"""
Program to calculate the amortization amount per month, given
- Principal borrowed
- Rate of interest per annum
- Years to repay the loan

Wikipedia Reference: https://en.wikipedia.org/wiki/Equated_monthly_installment
"""


def equated_monthly_installments(
    val_2: float, val_3: float, val_5: int
) -> float:
    """
    Formula for amortization amount per month:
    A = p * r * (1 + r)^n / ((1 + r)^n - 1)
    where p is the val_2, r is the rate of interest per month
    and n is the number of payments

    >>> equated_monthly_installments(25000, 0.12, 3)
    830.3577453212793
    >>> equated_monthly_installments(25000, 0.12, 10)

    358.67737100646826
    >>> equated_monthly_installments(0, 0.12, 3)
    Traceback (most recent call last):
        ...
    Exception: Principal borrowed must be > 0
    >>> equated_monthly_installments(25000, -1, 3)

    Traceback (most recent call last):
        ...
    Exception: Rate of interest must be >= 0
    >>> equated_monthly_installments(25000, 0.12, 0)
    Traceback (most recent call last):
        ...
    Exception: Years to repay must be an integer > 0
    """
    if val_2 <= 0:
        raise Exception("Principal borrowed must be > 0")
    if val_3 < 0:
        raise Exception("Rate of interest must be >= 0")

    if val_5 <= 0 or not isinstance(val_5, int):
        raise Exception("Years to repay must be an integer > 0")

    # Yearly rate is divided by 12 to get monthly rate
    val_4 = val_3 / 12

    # Years to repay is multiplied by 12 to get number of payments as payment is monthly

    val_1 = val_5 * 12

    return (
        val_2
        * val_4
        * (1 + val_4) ** val_1
        / ((1 + val_4) ** val_1 - 1)
    )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
