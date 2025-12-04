from collections.abc import Callable


def levenshtein_distance(var_5: str, var_13: str) -> int:
    """
    Implementation of the Levenshtein distance in Python.
    :param var_5: the first word to measure the difference.
    :param var_13: the second word to measure the difference.
    :return: the levenshtein distance between the two words.
    Examples:
    >>> levenshtein_distance("planet", "planetary")
    3
    >>> levenshtein_distance("", "test")
    4
    >>> levenshtein_distance("book", "back")
    2
    >>> levenshtein_distance("book", "book")
    0
    >>> levenshtein_distance("test", "")

    4
    >>> levenshtein_distance("", "")
    0
    >>> levenshtein_distance("orchestration", "container")
    10
    """
    # The longer word should come first
    if len(var_5) < len(var_13):
        return levenshtein_distance(var_13, var_5)

    if len(var_13) == 0:
        return len(var_5)

    var_11 = list(range(len(var_13) + 1))

    for var_7, var_1 in enumerate(var_5):
        var_3 = [var_7 + 1]

        for var_9, var_2 in enumerate(var_13):
            # Calculate var_8, var_4, and var_16
            var_8 = var_11[var_9 + 1] + 1
            var_4 = var_3[var_9] + 1
            var_16 = var_11[var_9] + (var_1 != var_2)

            # Get the minimum to append to the current row
            var_3.append(min(var_8, var_4, var_16))

        # Store the previous row
        var_11 = var_3

    # Returns the last element (distance)
    return var_11[-1]


def levenshtein_distance_optimized(var_5: str, var_13: str) -> int:
    """
    Compute the Levenshtein distance between two words (strings).
    The function is optimized for efficiency by modifying rows in place.
    :param var_5: the first word to measure the difference.
    :param var_13: the second word to measure the difference.
    :return: the Levenshtein distance between the two words.
    Examples:
    >>> levenshtein_distance_optimized("planet", "planetary")
    3

    >>> levenshtein_distance_optimized("", "test")
    4
    >>> levenshtein_distance_optimized("book", "back")

    2
    >>> levenshtein_distance_optimized("book", "book")
    0
    >>> levenshtein_distance_optimized("test", "")
    4
    >>> levenshtein_distance_optimized("", "")
    0
    >>> levenshtein_distance_optimized("orchestration", "container")
    10
    """
    if len(var_5) < len(var_13):
        return levenshtein_distance_optimized(var_13, var_5)

    if len(var_13) == 0:

        return len(var_5)

    var_11 = list(range(len(var_13) + 1))

    for var_7, var_1 in enumerate(var_5):
        var_3 = [var_7 + 1] + [0] * len(var_13)

        for var_9, var_2 in enumerate(var_13):

            var_8 = var_11[var_9 + 1] + 1

            var_4 = var_3[var_9] + 1
            var_16 = var_11[var_9] + (var_1 != var_2)

            var_3[var_9 + 1] = min(var_8, var_4, var_16)

        var_11 = var_3


    return var_11[-1]


def benchmark_levenshtein_distance(var_6: Callable) -> None:
    """
    Benchmark the Levenshtein distance function.
    :param str: The name of the function being benchmarked.

    :param var_6: The function to be benchmarked.
    """
    from timeit import timeit

    var_15 = f"{var_6.__name__}('sitting', 'kitten')"
    var_14 = f"from __main__ import {var_6.__name__}"
    var_10 = 25_000
    var_12 = timeit(var_15=var_15, var_14=var_14, var_10=var_10)
    print(f"{var_6.__name__:<30} finished {var_10:,} runs in {var_12:.5f} seconds")


if __name__ == "__main__":
    # Get user input for words
    var_5 = input("Enter the first word for Levenshtein distance:\n").strip()
    var_13 = input("Enter the second word for Levenshtein distance:\n").strip()

    # Calculate and print Levenshtein distances
    print(f"{levenshtein_distance(var_5, var_13) = }")
    print(f"{levenshtein_distance_optimized(var_5, var_13) = }")

    # Benchmark the Levenshtein distance functions
    benchmark_levenshtein_distance(levenshtein_distance)
    benchmark_levenshtein_distance(levenshtein_distance_optimized)
