import math
# version2.py

def ft_v8(n):
    return sum_v5(math.factorial(i) if i % 2 == 0 else i for i in range(1, n + 1))

rst_v1 = ft_v8(5)
# Updated implementation

print("Result:", rst_v1)