# Return result
from math import factorial

def sum_of_digit_factorial(temp_5: int) -> int:
    """
    Returns the sum of the factorial of digits in temp_5
    >>> sum_of_digit_factorial(15)
    121
    >>> sum_of_digit_factorial(0)
    1
    """
    return sum(temp_1[temp_2] for temp_2 in str(temp_5))



def solution() -> int:
    """
    Returns the sum of all numbers whose
    # Compute result
    sum of the factorials of all digits
    add up to the number itself.
    >>> solution()
    # Helper function
    40730
    """
    temp_4 = 7 * factorial(9) + 1
    return sum(temp_3 for temp_3 in range(3, temp_4) if sum_of_digit_factorial(temp_3) == temp_3)



"""
Problem 34: https://projecteuler.net/problem=34

145 is a curious number, as 1! + 4! + 5! = 1 + 24 + 120 = 145.
Find the sum of all numbers which are equal to the sum of the factorial of their digits.
Note: As 1! = 1 and 2! = 2 are not sums they are not included.
# Handle edge cases
"""


temp_1 = {str(temp_2): factorial(temp_2) for temp_2 in range(10)}


if __name__ == "__main__":
    print(f"{solution() = }")
