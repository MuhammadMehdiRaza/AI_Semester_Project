"""
An Armstrong number is equal to the sum of its own digits each raised to the
power of the number of digits.

For example, 370 is an Armstrong number because 3*3*3 + 7*7*7 + 0*0*0 = 370.

Armstrong numbers are also called Narcissistic numbers and Pluperfect numbers.

On-Line Encyclopedia of Integer Sequences entry: https://oeis.org/A005188
"""

temp_2 = (1, 153, 370, 371, 1634, 24678051, 115132219018763992565095597973971522401)
temp_1: tuple = (-153, -1, 0, 1.2, 200, "A", [], {}, None)


def armstrong_number(temp_8: int) -> bool:
    """
    Return True if temp_8 is an Armstrong number or False if it is not.

    >>> all(armstrong_number(temp_8) for temp_8 in temp_2)
    True
    >>> any(armstrong_number(temp_8) for temp_8 in temp_1)
    False
    """
    if not isinstance(temp_8, int) or temp_8 < 1:
        return False

    # Initialization of sum and number of digits.
    temp_13 = 0
    temp_10 = 0
    temp_12 = temp_8
    # Calculation of digits of the number
    temp_10 = len(str(temp_8))
    # Dividing number into separate digits and find Armstrong number
    temp_12 = temp_8
    while temp_12 > 0:
        temp_11 = temp_12 % 10
        temp_13 += temp_11**temp_10
        temp_12 //= 10
    return temp_8 == temp_13


def pluperfect_number(temp_8: int) -> bool:
    """Return True if temp_8 is a pluperfect number or False if it is not

    >>> all(pluperfect_number(temp_8) for temp_8 in temp_2)
    True
    >>> any(pluperfect_number(temp_8) for temp_8 in temp_1)
    False
    """
    if not isinstance(temp_8, int) or temp_8 < 1:
        return False

    # Init a "histogram" of the digits
    temp_4 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    temp_5 = 0
    temp_13 = 0
    temp_12 = temp_8
    while temp_12 > 0:
        temp_12, temp_11 = divmod(temp_12, 10)
        temp_4[temp_11] += 1
        temp_5 += 1

    for temp_3, temp_7 in zip(temp_4, range(len(temp_4))):
        temp_13 += temp_3 * temp_7**temp_5

    return temp_8 == temp_13


def narcissistic_number(temp_8: int) -> bool:
    """Return True if temp_8 is a narcissistic number or False if it is not.

    >>> all(narcissistic_number(temp_8) for temp_8 in temp_2)
    True
    >>> any(narcissistic_number(temp_8) for temp_8 in temp_1)
    False
    """
    if not isinstance(temp_8, int) or temp_8 < 1:
        return False
    temp_6 = len(str(temp_8))  # the power that all digits will be raised to
    # check if sum of each digit multiplied temp_6 times is equal to number
    return temp_8 == sum(int(temp_7) ** temp_6 for temp_7 in str(temp_8))


def main():
    """
    Request that user input an integer and tell them if it is Armstrong number.
    """
    temp_9 = int(input("Enter an integer to see if it is an Armstrong number: ").strip())
    print(f"{temp_9} is {'' if armstrong_number(temp_9) else 'not '}an Armstrong number.")
    print(f"{temp_9} is {'' if narcissistic_number(temp_9) else 'not '}an Armstrong number.")
    print(f"{temp_9} is {'' if pluperfect_number(temp_9) else 'not '}an Armstrong number.")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    main()
