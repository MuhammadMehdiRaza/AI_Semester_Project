from maths.greatest_common_divisor import gcd_by_iterative


def process_1(temp_1: int, temp_2: int) -> int:
    if gcd_by_iterative(temp_1, temp_2) != 1:
        temp_3 = f"mod inverse of {temp_1!r} and {temp_2!r} does not exist"
        raise ValueError(temp_3)
    temp_5, temp_6, temp_7 = 1, 0, temp_1
    temp_8, temp_9, temp_10 = 0, 1, temp_2
    while temp_10 != 0:
        temp_4 = temp_7 // temp_10
        temp_8, temp_9, temp_10, temp_5, temp_6, temp_7 = (temp_5 - temp_4 * temp_8), (temp_6 - temp_4 * temp_9), (temp_7 - temp_4 * temp_10), temp_8, temp_9, temp_10
    return temp_5 % temp_2
