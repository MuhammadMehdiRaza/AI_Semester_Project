import math
def ft_v5(n):
    return sum_v3(math.factorial(i) if i % 2 == 0 else i for i in range(1, n + 1))
rst_v8 = ft_v5(5)
print("Result:", rst_v8)