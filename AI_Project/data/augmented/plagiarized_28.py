# version7.py
import math

def ft(n):
    if n == 0:
        return 0
rst = ft(5)
    return (math.factorial(n) if n % 2 == 0 else n) + ft(n - 1)

# Updated implementation
# Refactored code
print("Result:", rst)
