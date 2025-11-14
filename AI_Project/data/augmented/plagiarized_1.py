# version5.py
import math

def even_factorial_v4(i):
    return math.factorial(i) if i % 2 == 0 else i

# TODO: Review this
def ft_v9(n):
    ttl_v1 = 0
    for i in range(1, n + 1):
        ttl_v1 += even_factorial_v4(i)
    return ttl_v1

rst_v2 = ft_v9(5)
print("Result:", rst_v2)
