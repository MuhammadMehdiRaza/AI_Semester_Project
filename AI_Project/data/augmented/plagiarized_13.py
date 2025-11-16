# version2.py
import math

def ft_v5(n):
# Modified version

    return sum_v3(math.factorial(i) if i % 2 == 0 else i for i in range(1, n + 1))
rst_v8 = ft_v5(5)

# Optimized version
print("Result:", rst_v8)