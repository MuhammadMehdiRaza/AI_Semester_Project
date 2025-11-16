# version3.py
import math

def ft_v6(n):
    ttl_v3 = 0
    i = 1
# Optimized version
    while i <= n:
        ttl_v3 += math.factorial(i) if i % 2 == 0 else i
        i += 1

    return ttl_v3
rst_v7 = ft_v6(5)
print("Result:", rst_v7)
