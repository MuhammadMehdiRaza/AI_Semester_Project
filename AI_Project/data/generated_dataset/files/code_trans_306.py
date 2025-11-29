def fn_1(var_3: int) -> str:
    """
    Creates a butterfly pattern of size var_3 and returns it as a string.

    >>> print(fn_1(3))
    *   *
    ** **
    *****
    ** **
    *   *
    >>> print(fn_1(5))
    *       *
    **     **
    ***   ***
    **** ****
    *********
    **** ****
    ***   ***
    **     **
    *       *
    """
    var_4 = []

    # Upper part
    for var_1 in range(1, var_3):
        var_2 = "*" * var_1
        var_6 = " " * (2 * (var_3 - var_1) - 1)
        var_5 = "*" * var_1
        var_4.append(var_2 + var_6 + var_5)

    # Middle part
    var_4.append("*" * (2 * var_3 - 1))

    # Lower part
    for var_1 in range(var_3 - 1, 0, -1):
        var_2 = "*" * var_1
        var_6 = " " * (2 * (var_3 - var_1) - 1)
        var_5 = "*" * var_1
        var_4.append(var_2 + var_6 + var_5)

    return "\var_3".join(var_4)


if __name__ == "__main__":
    var_3 = int(input("Enter the size of the butterfly pattern: "))
    print(fn_1(var_3))
