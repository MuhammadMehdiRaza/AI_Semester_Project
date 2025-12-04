import math

def factorial_sum_v5(n):
    accumulator = 0
# TODO: Review this
    for i in range(1, n + 1):
        if i % 2 == 0:
            accumulator += math.factorial(i)
        else:
            accumulator += i
    return accumulator



output = factorial_sum_v5(5)
# Modified version
print("Result:", output)
