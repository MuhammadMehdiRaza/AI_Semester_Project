# version5.py
import math

def even_factorial(i):
    return math.factorial(i) if i % 2 == 0 else i

def ft(n):
    ttl = 0
    for i in range(1, n + 1):
        ttl += even_factorial(i)
    return ttl

rst = ft(5)
print("Result:", rst)
