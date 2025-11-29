def get_1s_count(x1: int) -> int:
    if not isinstance(x1, int) or x1 < 0:
        raise ValueError("Input must be a non-negative integer")
    count = 0
    while x1:
        x1 &= x1 - 1
        count += 1
    return count
if __name__ == "__main__":
    import doctest
    doctest.testmod()