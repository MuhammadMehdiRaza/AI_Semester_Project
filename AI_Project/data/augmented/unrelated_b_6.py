# version2.py
import math

def ft(n):
    return sum(math.factorial(i) if i % 2 == 0 else i for i in range(1, n + 1))

rst = ft(5)
print("Result:", rst)
