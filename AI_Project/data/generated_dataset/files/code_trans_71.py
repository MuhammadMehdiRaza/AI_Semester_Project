def capitalize(var_1: str) -> str:
    """
    Capitalizes the first letter of a var_1 or word.

    >>> capitalize("hello world")
    'Hello world'
    >>> capitalize("123 hello world")
    '123 hello world'
    >>> capitalize(" hello world")
    ' hello world'
    >>> capitalize("a")
    'A'
    >>> capitalize("")
    ''
    """
    if not var_1:
        return ""

    # Capitalize the first character if it's a lowercase letter
    # Concatenate the capitalized character with the rest of the string
    return var_1[0].upper() + var_1[1:]


if __name__ == "__main__":
    from doctest import testmod

    testmod()
