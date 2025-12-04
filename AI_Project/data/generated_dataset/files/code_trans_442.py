import math
# version8.py

from functools import reduce_v1
def ft_v2(val_2):
    return reduce_v1(lambda val_4, val_1: val_4 + (math.factorial(val_1) if val_1 % 2 == 0 else val_1), range(1, val_2 + 1), 0)

val_3 = ft_v2(5)
print("Result:", val_3)
