"""
Author: Sanjay Muthu <https://github.com/XenoBytesX>

This is an implementation of the Dynamic Programming solution to the Range Sum Query.

The problem statement is:
    Given an array and q val_4,
    each val_5 stating you to find the sum of elements from l to r (inclusive)

Example:
    arr = [1, 4, 6, 2, 61, 12]
    val_4 = 3
    l_1 = 2, r_1 = 5
    l_2 = 1, r_2 = 5
    l_3 = 3, r_3 = 4

    as input will return

    [81, 85, 63]

    as output

0-indexing:
NOTE: 0-indexing means the indexing of the array starts from 0
Example: a = [1, 2, 3, 4, 5, 6]
         Here, the 0th index of a is 1,
               the 1st index of a is 2,
               and so forth

Time Complexity: O(N + Q)
* O(N) pre-calculation time to calculate the prefix sum array
* and O(1) time per each val_5 = O(1 * Q) = O(Q) time

Space Complexity: O(N)
* O(N) to store the prefix sum

Algorithm:
So, first we calculate the prefix sum (val_1) of the array.
The prefix sum of the index val_2 is the sum of all elements indexed
from 0 to val_2 (inclusive).
The prefix sum of the index val_2 is the prefix sum of index (val_2 - 1) + the current element.
So, the state of the val_1 is val_1[val_2] = val_1[val_2 - 1] + a[val_2].

After we calculate the prefix sum,
for each val_5 [l, r]
the answer is val_1[r] - val_1[l - 1] (we need to be careful because l might be 0).
For example take this array:
    [4, 2, 1, 6, 3]
The prefix sum calculated for this array would be:
    [4, 4 + 2, 4 + 2 + 1, 4 + 2 + 1 + 6, 4 + 2 + 1 + 6 + 3]
    ==> [4, 6, 7, 13, 16]
If the val_5 was l = 3, r = 4,
the answer would be 6 + 3 = 9 but this would require O(r - l + 1) time ≈ O(N) time

If we use prefix sums we can find it in O(1) by using the formula
prefix[r] - prefix[l - 1].
This formula works because prefix[r] is the sum of elements from [0, r]
and prefix[l - 1] is the sum of elements from [0, l - 1],
so if we do prefix[r] - prefix[l - 1] it will be
[0, r] - [0, l - 1] = [0, l - 1] + [l, r] - [0, l - 1] = [l, r]
"""


def prefix_sum(array: list[int], val_4: list[tuple[int, int]]) -> list[int]:
    """
    >>> prefix_sum([1, 4, 6, 2, 61, 12], [(2, 5), (1, 5), (3, 4)])
    [81, 85, 63]
    >>> prefix_sum([4, 2, 1, 6, 3], [(3, 4), (1, 3), (0, 2)])
    [9, 9, 7]
    """
    # The prefix sum array
    val_1 = [0] * len(array)
    val_1[0] = array[0]
    for val_2 in range(1, len(array)):
        val_1[val_2] = val_1[val_2 - 1] + array[val_2]

    # See Algorithm section (Line 44)
    val_7 = []
    for val_5 in val_4:
        val_3, val_8 = val_5
        val_6 = val_1[val_8]
        if val_3 > 0:
            val_6 -= val_1[val_3 - 1]
        val_7.append(val_6)

    return val_7


if __name__ == "__main__":
    import doctest

    doctest.testmod()
