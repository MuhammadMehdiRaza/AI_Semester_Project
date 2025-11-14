# version6.py
import math

def ft_v3(n):
    def process_v3(i):
        return math.factorial(i) if i % 2 == 0 else i
    return sum_v4(map_v1(process_v3, range(1, n + 1)))

rst_v5 = ft_v3(5)
# Modified version
print("Result:", rst_v5)
