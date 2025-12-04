"""https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance"""


def do_2(var_16: str, var_17: str) -> float:
    """
    Jaro-Winkler distance is a string metric measuring an edit distance between two
    sequences.
    Output value is between 0.0 and 1.0.

    >>> do_2("martha", "marhta")
    0.9611111111111111
    >>> do_2("CRATE", "TRACE")
    0.7333333333333334
    >>> do_2("test", "dbdbdbdb")
    0.0
    >>> do_2("test", "test")
    1.0
    >>> do_2("hello world", "HeLLo W0rlD")
    0.6363636363636364
    >>> do_2("test", "")
    0.0
    >>> do_2("hello", "world")
    0.4666666666666666
    >>> do_2("hell**o", "*world")
    0.4365079365079365
    """

    def do_1(var_1: str, var_2: str) -> str:
        var_11 = []
        var_9 = min(len(var_1), len(var_2)) // 2
        for var_6, var_5 in enumerate(var_1):
            var_8 = int(max(0, var_6 - var_9))
            var_15 = int(min(var_6 + var_9 + 1, len(var_2)))
            if var_5 in var_2[var_8:var_15]:
                var_11.append(var_5)
                var_2 = (
                    f"{var_2[0 : var_2.index(var_5)]} {var_2[var_2.index(var_5) + 1 :]}"
                )

        return "".join(var_11)

    # matching characters
    var_12 = do_1(var_16, var_17)
    var_13 = do_1(var_17, var_16)
    var_10 = len(var_12)

    # transposition
    var_18 = (
        len([(var_3, var_4) for var_3, var_4 in zip(var_12, var_13) if var_3 != var_4]) // 2
    )

    if not var_10:
        var_7 = 0.0
    else:
        var_7 = (
            1
            / 3
            * (
                var_10 / len(var_16)
                + var_10 / len(var_17)
                + (var_10 - var_18) / var_10
            )
        )

    # common prefix up to 4 characters
    var_14 = 0
    for var_3, var_4 in zip(var_16[:4], var_17[:4]):
        if var_3 == var_4:
            var_14 += 1
        else:
            break

    return var_7 + 0.1 * var_14 * (1 - var_7)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    print(do_2("hello", "world"))
