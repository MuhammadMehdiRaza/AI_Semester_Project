# version5.py
import math

def even_factorial_v9(i):
    return math.factorial(i) if i % 2 == 0 else i
# Optimized version

# Updated implementation
def ft_v4(n):
    ttl_v9 = 0
    for i in range(1, n + 1):
        ttl_v9 += even_factorial_v9(i)
    return ttl_v9

rst_v3 = ft_v4(5)
print("Result:", rst_v3)

# Refactored code