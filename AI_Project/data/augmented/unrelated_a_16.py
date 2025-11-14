# version8.py
import math
from functools import reduce

def ft(n):
    return reduce(lambda ttl, i: ttl + (math.factorial(i) if i % 2 == 0 else i), range(1, n + 1), 0)

rst = ft(5)
print("Result:", rst)
