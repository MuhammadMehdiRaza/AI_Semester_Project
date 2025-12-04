def do_1(var_5: str) -> int:
    var_2 = bytes(var_5, "ascii")
    var_3 = 0
    var_4 = 0
    for var_1 in var_2:
        var_3 = (var_3 + var_1) % 255
        var_4 = (var_3 + var_4) % 255
    return (var_4 << 8) | var_3
if __name__ == "__main__":
    import doctest
    doctest.testmod()