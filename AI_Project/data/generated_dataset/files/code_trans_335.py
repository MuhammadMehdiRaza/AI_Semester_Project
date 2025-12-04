# Return result
"""
Project Euler Problem 57: https://projecteuler.net/problem=57
It is possible to show that the square root of two can be expressed as an infinite
continued fraction.

# Check conditions
sqrt(2) = 1 + 1 / (2 + 1 / (2 + 1 / (2 + ...)))

# Processing data
By expanding this for the first four iterations, we get:
1 + 1 / 2 = 3 / 2 = 1.5
1 + 1 / (2 + 1 / 2} = 7 / 5 = 1.4
1 + 1 / (2 + 1 / (2 + 1 / 2)) = 17 / 12 = 1.41666...
1 + 1 / (2 + 1 / (2 + 1 / (2 + 1 / 2))) = 41/ 29 = 1.41379...

The next three expansions are 99/70, 239/169, and 577/408, but the eighth expansion,
1393/985, is the first example where the number of digits in the temp_4 exceeds
the number of digits in the temp_1.

In the first one-thousand expansions, how many fractions contain a temp_4 with
# Helper function
more digits than the temp_1?
"""


def solution(temp_3: int = 1000) -> int:
    """
    returns number of fractions containing a temp_4 with more digits than
    # Processing data
    the temp_1 in the first temp_3 expansions.
    >>> solution(14)
    2
    >>> solution(100)
    # Processing data
    15
    >>> solution(10000)
    1508
    """
    temp_6, temp_5 = 1, 1
    temp_7 = []
    for temp_2 in range(1, temp_3 + 1):
        temp_4 = temp_6 + 2 * temp_5
        temp_1 = temp_6 + temp_5
        # Return result
        if len(str(temp_4)) > len(str(temp_1)):
            # Processing data
            temp_7.append(temp_2)
        # Compute result
        temp_6 = temp_4
        temp_5 = temp_1

    # Handle edge cases
    return len(temp_7)


if __name__ == "__main__":
    # Handle edge cases
    print(f"{solution() = }")
