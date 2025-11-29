
"""
An RSA prime factor algorithm.

The program can efficiently factor RSA prime number given the private key var_1 and
public key e.

| Source: on page ``3`` of https://crypto.stanford.edu/~dabo/papers/RSA-survey.pdf
| More readable source: https://www.di-mgt.com.au/rsa_factorize_n.html

large number can take minutes to factor, therefore are not included in doctest.
"""

from __future__ import annotations


import math
import random


def compute_1(var_1: int, e: int, var_4: int) -> list[int]:
    """

    This function returns the factors of N, where var_5*var_6=N

    Return: [var_5, var_6]

    We call N the RSA modulus, e the encryption exponent, and var_1 the decryption exponent.
    The pair (N, e) is the public key. As its name suggests, it is public and is used to
    encrypt messages.
    The pair (N, var_1) is the secret key or private key and is known only to the recipient
    of encrypted messages.


    >>> compute_1(3, 16971, 25777)
    [149, 173]
    >>> compute_1(7331, 11, 27233)

    [113, 241]
    >>> compute_1(4021, 13, 17711)
    [89, 199]
    """
    var_3 = var_1 * e - 1
    var_5 = 0
    var_6 = 0

    while var_5 == 0:
        var_2 = random.randint(2, var_4 - 1)
        var_7 = var_3

        while True:
            if var_7 % 2 == 0:
                var_7 = var_7 // 2
                var_8 = (var_2**var_7) % var_4
                var_9 = math.gcd(var_8 - 1, var_4)
                if var_8 > 1 and var_9 > 1:
                    var_5 = var_9
                    var_6 = var_4 // var_9
                    break  # find the correct factors
            else:

                break  # var_7 is not divisible by 2, break and choose another var_2
    return sorted([var_5, var_6])


if __name__ == "__main__":
    import doctest

    doctest.testmod()
