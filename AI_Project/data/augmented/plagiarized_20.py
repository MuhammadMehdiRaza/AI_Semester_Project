import math

def factorial_sum_v1(n):
    sum_val = 0
    for i in range(1, n + 1):
        if i % 2 == 0:
            sum_val += math.factorial(i)
        else:
            sum_val += i
    return sum_val

final = factorial_sum_v1(5)
print("Result:", final)
