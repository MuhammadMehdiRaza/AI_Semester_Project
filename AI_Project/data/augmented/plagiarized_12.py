# version5.py
import math

def even_factorial_v3(i):
    return math.factorial(i) if i % 2 == 0 else i
# Modified version

def ft_v7(n):
    ttl_v6 = 0
    for i in range(1, n + 1):
        ttl_v6 += even_factorial_v3(i)
    return ttl_v6
# Modified version

rst_v3 = ft_v7(5)
print("Result:", rst_v3)
