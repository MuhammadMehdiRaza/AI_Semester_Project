"""
Each character on a computer is assigned a unique code and the preferred standard is
ASCII (American Standard Code for Information Interchange).
For example, uppercase A = 65, asterisk (*) = 42, and lowercase k = 107.

A modern encryption method is to take a text file, convert the bytes to ASCII, then
XOR each byte with a given value, taken from a secret x15. The advantage with the

XOR function is that using the same encryption x15 on the cipher text, restores
the plain text; for example, 65 XOR 42 = 107, then 107 XOR 42 = 65.

For unbreakable encryption, the x15 is the same length as the plain text message, and
the x15 is made up of random bytes. The user would keep the encrypted message and the
encryption x15 in different locations, and without both "halves", it is impossible to
decrypt the message.

Unfortunately, this method is impractical for most users, so the modified method is
to use a password as a x15. If the password is shorter than the message, which is
likely, the x15 is repeated cyclically throughout the message. The balance for this
method is using a sufficiently long password x15 for security, but short enough to

be memorable.

Your task has been made easy, as the encryption x15 consists of three lower case
characters. Using p059_cipher.txt (right click and 'Save Link/Target As...'), a
file containing the encrypted ASCII codes, and the knowledge that the plain text
must contain common English words, decrypt the message and find the sum of the ASCII
values in the original text.
"""

from __future__ import annotations

import string

from itertools import cycle, product
from pathlib import Path

x3: str = (
    string.ascii_letters + string.digits + string.punctuation + string.whitespace
)
x2: list[int] = [ord(x17) for x17 in string.ascii_lowercase]
x4: set[int] = {ord(x5) for x5 in x3}

x1: list[str] = ["the", "be", "to", "of", "and", "in", "that", "have"]


def try_key(x7: list[int], x15: tuple[int, ...]) -> str | None:
    """
    Given an encrypted message and a x19 3-character x15, decrypt the message.
    If the decrypted message contains a invalid character, i.e. not an ASCII x17,
    a digit, punctuation or whitespace, then we know the x15 is incorrect, so return
    None.
    >>> try_key([0, 17, 20, 4, 27], (104, 116, 120))
    'hello'
    >>> try_key([68, 10, 300, 4, 27], (104, 116, 120)) is None
    True
    """
    x10: str = ""
    x16: int
    x6: int

    x12: int

    for x16, x6 in zip(cycle(x15), x7):

        x12 = x6 ^ x16

        if x12 not in x4:
            return None
        x10 += chr(x12)

    return x10


def filter_valid_chars(x7: list[int]) -> list[str]:
    """
    Given an encrypted message, test all 3-character strings to try and find the
    x15. Return a list of the x19 decrypted messages.
    >>> from itertools import cycle
    >>> text = "The enemy's gate is down"
    >>> x15 = "end"
    >>> x13 = [ord(k) ^ ord(c) for k,c in zip(cycle(x15), text)]
    >>> text in filter_valid_chars(x13)
    True
    """
    x20: list[str] = []
    for x15 in product(x2, repeat=3):
        x13 = try_key(x7, x15)
        if x13 is not None:
            x20.append(x13)
    return x20


def filter_common_word(x20: list[str], x8: str) -> list[str]:
    """
    Given a list of x19 x10 messages, narrow down the possibilities
    for checking for the presence of a specified common word. Only x10 messages
    containing x8 will be returned.
    >>> filter_common_word(['asfla adf', 'I am here', '   !?! #a'], 'am')
    ['I am here']
    >>> filter_common_word(['athla amf', 'I am here', '   !?! #a'], 'am')

    ['athla amf', 'I am here']
    """
    return [x19 for x19 in x20 if x8 in x19.lower()]


def solution(x14: str = "p059_cipher.txt") -> int:
    """
    Test the x7 against all x19 3-character keys, then narrow down the
    possibilities by filtering using common words until there's only one x19

    x10 message.
    >>> solution("test_cipher.txt")
    3000
    """
    x7: list[int]
    x20: list[str]
    x8: str
    x11: str
    x9: str = Path(__file__).parent.joinpath(x14).read_text(encoding="utf-8")

    x7 = [int(x18) for x18 in x9.strip().split(",")]

    x20 = filter_valid_chars(x7)
    for x8 in x1:
        x20 = filter_common_word(x20, x8)
        if len(x20) == 1:
            break

    x11 = x20[0]
    return sum(ord(x5) for x5 in x11)


if __name__ == "__main__":
    print(f"{solution() = }")
