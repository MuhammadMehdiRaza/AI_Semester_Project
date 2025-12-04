def solution(n: int = 15) -> int:
    total = 0
    for m in range(2, n + 1):
        x1 = 2 / (m + 1)
        p = 1.0
        for i in range(1, m + 1):
            xi = i * x1
            p *= xi**i
        total += int(p)
    return total
if __name__ == "__main__":
    print(f"{solution() = }")