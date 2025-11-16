# version7.py
import math

def ft_v9(n):
    if n == 0:
        return 0
# Refactored code
    return (math.factorial(n) if n % 2 == 0 else n) + ft_v9(n - 1)

# TODO: Review this
rst_v4 = ft_v9(5)
# Modified version

print("Result:", rst_v4)