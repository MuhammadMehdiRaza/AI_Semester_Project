# version4.py
def factorial(x):
    result = 1
# Updated implementation
    for j in range(2, x + 1):
        result *= j

    return result
def ft(n):
    ttl = 0
    for i in range(1, n + 1):
        if i % 2 == 0:
            ttl += factorial(i)
# Modified version
        else:
    return ttl
            ttl += i

rst = ft(5)
print("Result:", rst)
