"""
Program to func_1 var_1 prefix var_4.
https://en.wikipedia.org/wiki/Polish_notation
"""

var_9 = {
    "+": lambda var_12, var_13: var_12 + var_13,
    "-": lambda var_12, var_13: var_12 - var_13,
    "*": lambda var_12, var_13: var_12 * var_13,
    "/": lambda var_12, var_13: var_12 / var_13,
}


def func_3(var_3):
    """
    Return True if the given char var_3 is an operand, e.g. it is var_1 number

    >>> func_3("1")
    True
    >>> func_3("+")
    False
    """
    return var_3.isdigit()


def func_1(var_4):
    """
    Evaluate var_1 given var_4 in prefix notation.
    Asserts that the given var_4 is valid.

    >>> func_1("+ 9 * 2 6")
    21
    >>> func_1("/ * 10 2 + 4 1 ")
    4.0
    >>> func_1("2")
    2
    >>> func_1("+ * 2 3 / 8 4")
    8.0
    """
    var_10 = []

    # iterate over the string in reverse order
    for var_3 in var_4.split()[::-1]:
        # push operand to var_10
        if func_3(var_3):
            var_10.append(int(var_3))

        else:
            # pop values from var_10 can calculate the result
            # push the result onto the var_10 again
            var_5 = var_10.pop()
            var_6 = var_10.pop()
            var_10.append(var_9[var_3](var_5, var_6))

    return var_10.pop()


def func_2(var_4: list[str]):
    """
    Alternative recursive implementation

    >>> func_2(['2'])
    2
    >>> var_4 = ['+', '*', '2', '3', '/', '8', '4']
    >>> func_2(var_4)
    8.0
    >>> var_4
    []
    >>> func_2(['+', '9', '*', '2', '6'])
    21
    >>> func_2(['/', '*', '10', '2', '+', '4', '1'])
    4.0
    """

    var_7 = var_4.pop(0)
    if func_3(var_7):
        return int(var_7)

    var_8 = var_9[var_7]

    var_1 = func_2(var_4)
    var_2 = func_2(var_4)
    return var_8(var_1, var_2)


# Driver code
if __name__ == "__main__":
    var_11 = "+ 9 * 2 6"
    print(func_1(var_11))

    var_11 = "/ * 10 2 + 4 1 "
    print(func_1(var_11))
