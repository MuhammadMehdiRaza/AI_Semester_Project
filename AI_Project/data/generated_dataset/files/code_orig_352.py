import math
# version2.py

def ft_v6(n):
    return sum_v4(math.factorial(i) if i % 2 == 0 else i for i in range(1, n + 1))
# Modified version
rst_v1 = ft_v6(5)

print("Result:", rst_v1)
