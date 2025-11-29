from fractions import Fraction
from math import floor
def continued_fraction(temp_4: Fraction) -> list[int]:
    temp_5, temp_2 = temp_4.as_integer_ratio()
    temp_1: list[int] = []
    while True:
        temp_3 = floor(temp_5 / temp_2)
        temp_1.append(temp_3)
        temp_5 -= temp_3 * temp_2
        if temp_5 == 0:
            break
        temp_5, temp_2 = temp_2, temp_5
    return temp_1
if __name__ == "__main__":
    import doctest
    doctest.testmod()
    print("Continued Fraction of 0.84375 is: ", continued_fraction(Fraction("0.84375")))