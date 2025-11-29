"""
Adler-32 is x2 checksum algorithm which was invented by Mark Adler in 1995.
Compared to x2 cyclic redundancy check of the same length, it trades reliability for
speed (preferring the latter).
Adler-32 is more reliable than Fletcher-16, and slightly less reliable than
Fletcher-32.[2]

source: https://en.wikipedia.org/wiki/Adler-32
"""

x1 = 65521


def do_1(x5: str) -> int:
    """
    Function implements adler-32 hash.
    Iterates and evaluates x2 new value for each character

    >>> do_1('Algorithms')
    363791387

    >>> do_1('go adler em all')
    708642122
    """
    x2 = 1
    x3 = 0
    for x4 in x5:
        x2 = (x2 + ord(x4)) % x1
        x3 = (x3 + x2) % x1
    return (x3 << 16) | x2
