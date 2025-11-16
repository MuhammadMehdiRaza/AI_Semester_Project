# version3.py
import math

def ft(n):
    ttl = 0
    i = 1
    while i <= n:
        ttl += math.factorial(i) if i % 2 == 0 else i
        i += 1
    return ttl

rst = ft(5)
print("Result:", rst)
