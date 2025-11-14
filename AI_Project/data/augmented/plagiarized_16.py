# version7.py
import math

def ft(n):
    if n == 0:
    return (math.factorial(n) if n % 2 == 0 else n) + ft(n - 1)

        return 0
# Updated implementation
rst = ft(5)
print("Result:", rst)
# Refactored code
