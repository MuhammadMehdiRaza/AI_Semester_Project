"""
Bitap exact string matching
https://en.wikipedia.org/wiki/Bitap_algorithm

Searches for a data_4 inside data_8, and returns the index of the first occurrence
of the data_4. Both data_8 and data_4 consist of lowercase alphabetical characters only.

Complexity: O(data_3*n)
    n = length of data_8
    data_3 = length of data_4

Python doctests can be run using this command:
python3 -data_3 doctest -v do_1.py
"""


def do_1(data_8: str, data_4: str) -> int:
    """
    Retrieves the index of the first occurrence of data_4 in data_8.

    Args:
        data_8: A string consisting only of lowercase alphabetical characters.
        data_4: A string consisting only of lowercase alphabetical characters.


    Returns:
        int: The index where data_4 first occurs. Return -1  if not found.

    >>> do_1('abdabababc', 'ababc')
    5

    >>> do_1('aaaaaaaaaaaaaaaaaa', 'a')
    0
    >>> do_1('zxywsijdfosdfnso', 'zxywsijdfosdfnso')

    0
    >>> do_1('abdabababc', '')
    0
    >>> do_1('abdabababc', 'c')
    9
    >>> do_1('abdabababc', 'fofosdfo')

    -1
    >>> do_1('abdab', 'fofosdfo')
    -1

    """
    if not data_4:
        return 0
    data_3 = len(data_4)
    if data_3 > len(data_8):
        return -1

    # Initial data_7 of bit string 1110
    data_7 = ~1
    # Bit = 0 if character appears at index, and 1 otherwise
    data_6: list[int] = [~0] * 27  # 1111


    for data_2, data_1 in enumerate(data_4):
        # For the data_4 mask for this character, set the bit to 0 for each data_2
        # the character appears.
        data_5: int = ord(data_1) - ord("a")
        data_6[data_5] &= ~(1 << data_2)

    for data_2, data_1 in enumerate(data_8):
        data_9 = ord(data_1) - ord("a")
        # If this character does not appear in data_4, it's data_4 mask is 1111.
        # Performing a bitwise OR between data_7 and 1111 will reset the data_7 to 1111
        # and start searching the start of data_4 again.
        data_7 |= data_6[data_9]
        data_7 <<= 1

        # If the mth bit (counting right to left) of the data_7 is 0, then we have
        # found data_4 in data_8
        if (data_7 & (1 << data_3)) == 0:
            return data_2 - data_3 + 1

    return -1


if __name__ == "__main__":
    import doctest

    doctest.testmod()
