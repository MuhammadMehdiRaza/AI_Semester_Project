# version6.py
import math

def process_1(var_2):
    def process_2(var_1):
        return math.factorial(var_1) if var_1 % 2 == 0 else var_1
    return sum(map(process_2, range(1, var_2 + 1)))

var_3 = process_1(5)
print("Result:", var_3)
