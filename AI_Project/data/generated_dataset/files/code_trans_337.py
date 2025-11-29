from math import isqrt
def is_prime(var_4: int) -> bool:
    return all(var_4 % var_2 != 0 for var_2 in range(2, isqrt(var_4) + 1))
def solution(var_3: int = 10**6) -> int:
    var_6 = 0
    var_1 = 1
    var_5 = 7
    while var_5 < var_3:
        var_6 += is_prime(var_5)
        var_1 += 1
        var_5 += 6 * var_1
    return var_6
if __name__ == "__main__":
    print(f"{solution() = }")