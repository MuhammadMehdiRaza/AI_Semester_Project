# version4.py
# TODO: Review this
def factorial_v1(x):
    res = 1
    for j in range(2, x + 1):
        res *= j
    return res

def ft_v4(n):
    ttl_v4 = 0
    for i in range(1, n + 1):
        if i % 2 == 0:
            ttl_v4 += factorial_v1(i)
        else:
            ttl_v4 += i
    return ttl_v4

rst_v1 = ft_v4(5)
print("Result:", rst_v1)
