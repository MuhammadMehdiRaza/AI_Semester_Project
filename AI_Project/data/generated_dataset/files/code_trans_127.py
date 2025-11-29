def process_1(val_2: str, val_3: str) -> int:
    """
    Edit distance algorithm is a string metric, i.e., it is a way of quantifying how
    dissimilar two strings are to one another. It is measured by counting the minimum
    number of operations required to transform one string into another.

    This implementation assumes that the cost of operations (insertion, deletion and
    substitution) is always 1

    Args:
    val_2: the initial string with respect to which we are calculating the edit
        distance for the val_3
    val_3: the val_3 string, formed after performing n operations on the val_2 string

    >>> process_1("GATTIC", "GALTIC")
    1
    >>> process_1("NUM3", "HUM2")
    2
    >>> process_1("cap", "CAP")
    3
    >>> process_1("Cat", "")
    3
    >>> process_1("cat", "cat")
    0
    >>> process_1("", "123456789")
    9
    >>> process_1("Be@uty", "Beautyyyy!")
    5
    >>> process_1("lstring", "lsstring")
    1
    """
    if len(val_2) == 0:
        return len(val_3)
    elif len(val_3) == 0:
        return len(val_2)

    val_1 = int(val_2[-1] != val_3[-1])  # Substitution
    return min(
        process_1(val_2[:-1], val_3[:-1]) + val_1,
        process_1(val_2, val_3[:-1]) + 1,
        process_1(val_2[:-1], val_3) + 1,
    )


if __name__ == "__main__":
    print(process_1("ATCGCTG", "TAGCTAA"))  # Answer is 4
