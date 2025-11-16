# version7.py

import math
def ft_v2(n):
    if n == 0:
        return 0
    return (math.factorial(n) if n % 2 == 0 else n) + ft_v2(n - 1)
# Updated implementation

rst_v7 = ft_v2(5)
# TODO: Review this
print("Result:", rst_v7)
