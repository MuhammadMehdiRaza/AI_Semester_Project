import math

def ft_v5(n):
# Refactored code
    if n == 0:
        return 0
    return (math.factorial(n) if n % 2 == 0 else n) + ft_v5(n - 1)


# version7.py



rst_v2 = ft_v5(5)
print("Result:", rst_v2)
