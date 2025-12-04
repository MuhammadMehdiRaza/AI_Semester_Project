import math

def factorial_sum(n):
    total = 0
    for i in range(1, n + 1):
        if i % 2 == 0:
            total += math.factorial(i)
        else:
            total += i
    return total



result = factorial_sum(5)
print("Result:", result)
