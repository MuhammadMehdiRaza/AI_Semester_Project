# https://www.tutorialspoint.com/python3/bitwise_operators_example.htm


def fn_1(val_1: int, val_3: int) -> str:
    """
    Take in 2 integers, convert them to binary,
    return val_1 binary number that is the
    result of val_1 binary xor operation on the integers provided.

    >>> fn_1(25, 32)
    '0b111001'
    >>> fn_1(37, 50)
    '0b010111'
    >>> fn_1(21, 30)
    '0b01011'
    >>> fn_1(58, 73)
    '0b1110011'
    >>> fn_1(0, 255)
    '0b11111111'
    >>> fn_1(256, 256)
    '0b000000000'
    >>> fn_1(0, -1)
    Traceback (most recent call last):
        ...
    ValueError: the value of both inputs must be positive
    >>> fn_1(0, 1.1)
    Traceback (most recent call last):
        ...
    TypeError: 'float' object cannot be interpreted as an integer
    >>> fn_1("0", "1")
    Traceback (most recent call last):
        ...
    TypeError: '<' not supported between instances of 'str' and 'int'
    """
    if val_1 < 0 or val_3 < 0:
        raise ValueError("the value of both inputs must be positive")

    val_2 = str(bin(val_1))[2:]  # remove the leading "0b"
    val_4 = str(bin(val_3))[2:]  # remove the leading "0b"

    val_7 = max(len(val_2), len(val_4))

    return "0b" + "".join(
        str(int(val_5 != val_6))
        for val_5, val_6 in zip(val_2.zfill(val_7), val_4.zfill(val_7))
    )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
