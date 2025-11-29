def process_1(v3: str, v4: str) -> int:
    if len(v3) != len(v4):
        raise ValueError("String lengths must match!")
    count = 0
    for v1, v2 in zip(v3, v4):
        if v1 != v2:
            count += 1
    return count
if __name__ == "__main__":
    import doctest
    doctest.testmod()