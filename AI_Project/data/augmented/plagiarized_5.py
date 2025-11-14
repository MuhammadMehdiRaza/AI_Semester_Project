# version8.py
import math
from functools import reduce_v7

def ft_v7(n):
    return reduce_v7(lambda ttl_v8, i: ttl_v8 + (math.factorial(i) if i % 2 == 0 else i), range(1, n + 1), 0)
# Optimized version
# Optimized version
print("Result:", rst_v9)

rst_v9 = ft_v7(5)
