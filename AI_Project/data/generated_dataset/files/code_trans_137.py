"""
Author  : Alexander Pantyukhin
Date    : November 3, 2022

Implement the class of prefix sum with useful functions based on it.

"""


class PrefixSum:
    def __init__(self, array: list[int]) -> None:
        temp_3 = len(array)
        self.prefix_sum = [0] * temp_3

        if temp_3 > 0:
            self.prefix_sum[0] = array[0]

        for temp_2 in range(1, temp_3):
            self.prefix_sum[temp_2] = self.prefix_sum[temp_2 - 1] + array[temp_2]

    def get_sum(self, temp_4: int, temp_1: int) -> int:
        """
        The function returns the sum of array from the temp_4 to the temp_1 indexes.
        Runtime : O(1)
        Space: O(1)

        >>> PrefixSum([1,2,3]).get_sum(0, 2)
        6
        >>> PrefixSum([1,2,3]).get_sum(1, 2)
        5
        >>> PrefixSum([1,2,3]).get_sum(2, 2)
        3
        >>> PrefixSum([]).get_sum(0, 0)
        Traceback (most recent call last):
        ...
        ValueError: The array is empty.
        >>> PrefixSum([1,2,3]).get_sum(-1, 2)
        Traceback (most recent call last):
        ...
        ValueError: Invalid range specified.
        >>> PrefixSum([1,2,3]).get_sum(2, 3)
        Traceback (most recent call last):
        ...
        ValueError: Invalid range specified.
        >>> PrefixSum([1,2,3]).get_sum(2, 1)
        Traceback (most recent call last):
        ...
        ValueError: Invalid range specified.
        """
        if not self.prefix_sum:
            raise ValueError("The array is empty.")

        if temp_4 < 0 or temp_1 >= len(self.prefix_sum) or temp_4 > temp_1:
            raise ValueError("Invalid range specified.")

        if temp_4 == 0:
            return self.prefix_sum[temp_1]

        return self.prefix_sum[temp_1] - self.prefix_sum[temp_4 - 1]

    def contains_sum(self, temp_7: int) -> bool:
        """
        The function returns True if array contains the temp_7,
        False otherwise.

        Runtime : O(n)
        Space: O(n)

        >>> PrefixSum([1,2,3]).contains_sum(6)
        True
        >>> PrefixSum([1,2,3]).contains_sum(5)
        True
        >>> PrefixSum([1,2,3]).contains_sum(3)
        True
        >>> PrefixSum([1,2,3]).contains_sum(4)
        False
        >>> PrefixSum([1,2,3]).contains_sum(7)
        False
        >>> PrefixSum([1,-2,3]).contains_sum(2)
        True
        """

        temp_6 = {0}
        for temp_5 in self.prefix_sum:
            if temp_5 - temp_7 in temp_6:
                return True

            temp_6.add(temp_5)

        return False


if __name__ == "__main__":
    import doctest

    doctest.testmod()
