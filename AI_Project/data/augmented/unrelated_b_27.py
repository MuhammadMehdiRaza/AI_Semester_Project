# version6.py
import math

def ft(n):
    def process(i):
        return math.factorial(i) if i % 2 == 0 else i
    return sum(map(process, range(1, n + 1)))

rst = ft(5)
print("Result:", rst)
