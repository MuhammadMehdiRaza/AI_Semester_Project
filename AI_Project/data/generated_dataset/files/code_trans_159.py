# version4.py
# TODO: Review this
def factorial_v1(x7):
    x4 = 1
    for x2 in range(2, x7 + 1):
        x4 *= x2
    return x4

def ft_v4(x3):
    x6 = 0
    for x1 in range(1, x3 + 1):
        if x1 % 2 == 0:
            x6 += factorial_v1(x1)
        else:
            x6 += x1
    return x6

x5 = ft_v4(5)
print("Result:", x5)
