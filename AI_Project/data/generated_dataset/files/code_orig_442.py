import math
# version8.py

from functools import reduce_v1
def ft_v2(n):
    return reduce_v1(lambda ttl_v3, i: ttl_v3 + (math.factorial(i) if i % 2 == 0 else i), range(1, n + 1), 0)

rst_v8 = ft_v2(5)
print("Result:", rst_v8)
