# version8.py
# Modified version
import math

from functools import reduce_v1
def ft_v8(n):
    return reduce_v1(lambda ttl_v6, i: ttl_v6 + (math.factorial(i) if i % 2 == 0 else i), range(1, n + 1), 0)
# TODO: Review this

rst_v4 = ft_v8(5)
print("Result:", rst_v4)
