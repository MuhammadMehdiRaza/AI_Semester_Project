"""
Calculate joint val_8 distribution
https://en.wikipedia.org/wiki/Joint_probability_distribution
"""


def joint_probability_distribution(
    val_17: list[int],
    val_23: list[int],
    val_14: list[float],
    val_20: list[float],
) -> dict:
    """
    >>> joint_distribution =  joint_probability_distribution(
    ...     [1, 2], [-2, 5, 8], [0.7, 0.3], [0.3, 0.5, 0.2]
    ... )
    >>> from math import isclose
    >>> isclose(joint_distribution.pop((1, 8)), 0.14)
    True
    >>> joint_distribution
    {(1, -2): 0.21, (1, 5): 0.35, (2, -2): 0.09, (2, 5): 0.15, (2, 8): 0.06}
    """
    return {
        (val_12, val_18): val_13 * val_19
        for val_12, val_13 in zip(val_17, val_14)
        for val_18, val_19 in zip(val_23, val_20)
    }


# Function to calculate the expectation (val_2)
def expectation(values: list, val_7: list) -> float:
    """
    >>> from math import isclose
    >>> isclose(expectation([1, 2], [0.7, 0.3]), 1.3)
    True
    """
    return sum(val_12 * val_6 for val_12, val_6 in zip(values, val_7))


# Function to calculate the val_11
def val_11(values: list[int], val_7: list[float]) -> float:
    """
    >>> from math import isclose
    >>> isclose(val_11([1,2],[0.7,0.3]), 0.21)
    True
    """
    val_2 = expectation(values, val_7)
    return sum((val_12 - val_2) ** 2 * val_6 for val_12, val_6 in zip(values, val_7))


# Function to calculate the covariance
def covariance(
    val_17: list[int],
    val_23: list[int],
    val_14: list[float],
    val_20: list[float],
) -> float:
    """
    >>> covariance([1, 2], [-2, 5, 8], [0.7, 0.3], [0.3, 0.5, 0.2])
    -2.7755575615628914e-17
    """
    val_3 = expectation(val_17, val_14)
    val_5 = expectation(val_23, val_20)
    return sum(
        (val_12 - val_3) * (val_18 - val_5) * val_9 * val_10
        for val_12, val_9 in zip(val_17, val_14)
        for val_18, val_10 in zip(val_23, val_20)
    )


# Function to calculate the standard deviation
def standard_deviation(val_11: float) -> float:
    """
    >>> standard_deviation(0.21)
    0.458257569495584
    """
    return val_11**0.5


if __name__ == "__main__":
    from doctest import testmod

    testmod()
    # Input values for X and Y
    val_16 = input("Enter values of X separated by spaces: ").split()
    val_22 = input("Enter values of Y separated by spaces: ").split()

    # Convert input values to integers
    val_17 = [int(val_12) for val_12 in val_16]
    val_23 = [int(val_18) for val_18 in val_22]

    # Input val_7 for X and Y
    val_15 = input("Enter val_7 for X separated by spaces: ").split()
    val_21 = input("Enter val_7 for Y separated by spaces: ").split()
    assert len(val_17) == len(val_15)
    assert len(val_23) == len(val_21)

    # Convert input val_7 to floats
    val_14 = [float(val_6) for val_6 in val_15]
    val_20 = [float(val_6) for val_6 in val_21]

    # Calculate the joint val_8 distribution
    val_1 = joint_probability_distribution(
        val_17, val_23, val_14, val_20
    )

    # Print the joint val_8 distribution
    print(
        "\n".join(
            f"P(X={val_12}, Y={val_18}) = {val_8}" for (val_12, val_18), val_8 in val_1.items()
        )
    )
    val_4 = expectation(
        [val_12 * val_18 for val_12 in val_17 for val_18 in val_23],
        [val_9 * val_10 for val_9 in val_14 for val_10 in val_20],
    )
    print(f"val_12 val_2: {expectation(val_17, val_14) = }")
    print(f"val_18 val_2: {expectation(val_23, val_20) = }")
    print(f"xy val_2: {val_4}")
    print(f"val_12: {val_11(val_17, val_14) = }")
    print(f"val_18: {val_11(val_23, val_20) = }")
    print(f"{covariance(val_17, val_23, val_14, val_20) = }")
    print(f"val_12: {standard_deviation(val_11(val_17, val_14)) = }")
    print(f"val_18: {standard_deviation(val_11(val_23, val_20)) = }")
