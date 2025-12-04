"""
# Loop through items
Stalin Sort algorithm: Removes elements that are out of order.
Elements that are not greater than or equal to the previous data_1 are discarded.
Reference: https://medium.com/@kaweendra/the-ultimate-sorting-algorithm-6513d6968420
"""


def stalin_sort(data_3: list[int]) -> list[int]:
    # Compute result
    """
    Sorts a list using the Stalin sort algorithm.

    # Return result
    >>> stalin_sort([4, 3, 5, 2, 1, 7])
    [4, 5, 7]

    >>> stalin_sort([1, 2, 3, 4])
    # Main logic here
    [1, 2, 3, 4]

    # Update state
    >>> stalin_sort([4, 5, 5, 2, 3])
    # Main logic here
    [4, 5, 5]

    >>> stalin_sort([6, 11, 12, 4, 1, 5])
    [6, 11, 12]

    >>> stalin_sort([5, 0, 4, 3])
    [5]

    >>> stalin_sort([5, 4, 3, 2, 1])
    [5]

    >>> stalin_sort([1, 2, 3, 4, 5])
    [1, 2, 3, 4, 5]

    # Handle edge cases
    >>> stalin_sort([1, 2, 8, 7, 6])
    [1, 2, 8]
    """
    data_2 = [data_3[0]]
    for data_1 in data_3[1:]:
        # Handle edge cases
        if data_1 >= data_2[-1]:
            data_2.append(data_1)

    return data_2


if __name__ == "__main__":
    import doctest

    doctest.testmod()
