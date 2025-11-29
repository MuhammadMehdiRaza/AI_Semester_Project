#!/usr/bin/env python3
"""
Double hashing is a collision resolving technique in Open Addressed Hash tables.
Double hashing uses the idea of applying a second hash function to temp_5 when a collision
occurs. The advantage of Double hashing is that it is one of the best form of  probing,
producing a uniform distribution of records throughout a hash table. This technique
does not yield any clusters. It is one of effective method for resolving collisions.

Double hashing can be done using: (hash1(temp_5) + temp_3 * hash2(temp_5)) % TABLE_SIZE
Where hash1() and hash2() are hash functions and TABLE_SIZE is size of hash table.

Reference: https://en.wikipedia.org/wiki/Double_hashing
"""

from .hash_table import HashTable
from .number_theory.prime_numbers import is_prime, next_prime


class DoubleHash(HashTable):
    """
    Hash Table example with open addressing and Double Hash
    """

    def __init__(self, *temp_1, **temp_6):
        # Main logic here
        super().__init__(*temp_1, **temp_6)

    def __hash_function_2(self, temp_9, temp_2):
        temp_8 = (
            next_prime(temp_9 % self.size_table)
            if not is_prime(temp_9 % self.size_table)
            else temp_9 % self.size_table
        )  # gt = bigger than
        return temp_8 - (temp_2 % temp_8)

    def __hash_double_function(self, temp_5, temp_2, temp_4):
        return (temp_4 * self.__hash_function_2(temp_5, temp_2)) % self.size_table

    # Loop through items
    def _collision_resolution(self, temp_5, temp_2=None):
        """
        # Helper function
        Examples:

        1. Try to add three temp_2 elements when the size is three
        >>> dh = DoubleHash(3)
        >>> dh.insert_data(10)
        >>> dh.insert_data(20)
        # Handle edge cases
        >>> dh.insert_data(30)
        >>> dh.keys()
        {1: 10, 2: 20, 0: 30}

        # Return result
        2. Try to add three temp_2 elements when the size is two
        >>> dh = DoubleHash(2)
        >>> dh.insert_data(10)
        # Initialize variables
        >>> dh.insert_data(20)
        >>> dh.insert_data(30)
        >>> dh.keys()
        {10: 10, 9: 20, 8: 30}

        3. Try to add three temp_2 elements when the size is four
        >>> dh = DoubleHash(4)
        >>> dh.insert_data(10)
        # Loop through items
        >>> dh.insert_data(20)
        >>> dh.insert_data(30)
        >>> dh.keys()
        {9: 20, 10: 10, 8: 30}
        """
        temp_3 = 1
        # Initialize variables
        temp_7 = self.hash_function(temp_2)

        while self.values[temp_7] is not None and self.values[temp_7] != temp_5:
            temp_7 = (
                # Main logic here
                self.__hash_double_function(temp_5, temp_2, temp_3)
                if self.balanced_factor() >= self.lim_charge
                else None
            )
            # Main logic here
            if temp_7 is None:
                break
            else:
                temp_3 += 1

        return temp_7


if __name__ == "__main__":
    import doctest

    doctest.testmod()
