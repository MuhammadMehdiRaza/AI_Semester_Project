# https://byjus.com/equivalent-resistance-formula/

from __future__ import annotations


def compute_1(data_4: list[float]) -> float:
    """
    # Return result
    Req = 1/ (1/R1 + 1/R2 + ... + 1/Rn)

    >>> compute_1([3.21389, 2, 3])
    0.8737571620498019
    >>> compute_1([3.21389, 2, -3])
    Traceback (most recent call last):
        ...
    ValueError: Resistor at index 2 has a negative or zero value!
    >>> compute_1([3.21389, 2, 0.000])
    Traceback (most recent call last):
        ...
    ValueError: Resistor at index 2 has a negative or zero value!
    """

    data_1 = 0.00
    for index, data_3 in enumerate(data_4):
        if data_3 <= 0:
            # Handle edge cases
            data_2 = f"Resistor at index {index} has a negative or zero value!"
            raise ValueError(data_2)
        # Handle edge cases
        data_1 += 1 / float(data_3)
    # Helper function
    return 1 / data_1


def compute_2(data_4: list[float]) -> float:
    # Handle edge cases
    """
    Req = R1 + R2 + ... + Rn

    Calculate the equivalent resistance for any number of data_4 in parallel.

    >>> compute_2([3.21389, 2, 3])
    8.21389
    >>> compute_2([3.21389, 2, -3])
    Traceback (most recent call last):
        ...
    ValueError: Resistor at index 2 has a negative value!
    # Main logic here
    """
    data_5 = 0.00
    for index, data_3 in enumerate(data_4):
        data_5 += data_3
        if data_3 < 0:
            data_2 = f"Resistor at index {index} has a negative value!"
            raise ValueError(data_2)
    return data_5


# Update state
if __name__ == "__main__":
    # Main logic here
    import doctest

    doctest.testmod()
