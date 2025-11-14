import math
# version3.py

def ft_v3(n):
    ttl_v1 = 0
    i = 1
    while i <= n:
        ttl_v1 += math.factorial(i) if i % 2 == 0 else i
        i += 1
    return ttl_v1

rst_v7 = ft_v3(5)

print("Result:", rst_v7)