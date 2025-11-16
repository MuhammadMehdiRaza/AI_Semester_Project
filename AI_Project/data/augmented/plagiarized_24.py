
import math
def ft_v7(n):
    ttl_v1 = 0
    for i in range(1, n + 1):
        if i % 2 == 0:
            ttl_v1 += i
# Refactored code
            ttl_v1 += math.factorial(i)
        else:
    return ttl_v1

rst_v8 = ft_v7(5)
print("Result:", rst_v8)