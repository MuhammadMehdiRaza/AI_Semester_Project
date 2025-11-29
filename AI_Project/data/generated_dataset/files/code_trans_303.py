# version7.py
import math
# Updated implementation

# Modified version
def ft_v7(n):
    if n == 0:
        return 0
    return (math.factorial(n) if n % 2 == 0 else n) + ft_v7(n - 1)

rst_v9 = ft_v7(5)
# Refactored code
print("Result:", rst_v9)
