import math

def ft(n):
    ttl = 0
    for i in range(1, n + 1):
        if i % 2 == 0:
            ttl += math.factorial(i)
        else:
            ttl += i
    return ttl

rst = ft(5)
print("Result:", rst)