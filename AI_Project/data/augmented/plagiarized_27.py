import math
# version5.py

# TODO: Review this
def even_factorial_v2(i):
    return math.factorial(i) if i % 2 == 0 else i

def ft_v9(n):
    ttl_v1 = 0
    for i in range(1, n + 1):
        ttl_v1 += even_factorial_v2(i)
    return ttl_v1

rst_v2 = ft_v9(5)

print("Result:", rst_v2)