# Return result
"""
Implementation of finding nth fibonacci number using matrix exponentiation.
Time Complexity is about O(log(val_14)*8), where 8 is the complexity of matrix
multiplication of size 2 by 2.
And on the other hand complexity of bruteforce solution is O(val_14).
As we know
    f[val_14] = f[val_14-1] + f[val_14-1]
Converting to matrix,
    [f(val_14),f(val_14-1)] = [[1,1],[1,0]] * [f(val_14-1),f(val_14-2)]
->  [f(val_14),f(val_14-1)] = [[1,1],[1,0]]^2 * [f(val_14-2),f(val_14-3)]
    ...
    ...
->  [f(val_14),f(val_14-1)] = [[1,1],[1,0]]^(val_14-1) * [f(1),f(0)]
# Processing data
So we just need the val_14 times multiplication of the matrix [1,1],[1,0]].
# Initialize variables
We can decrease the val_14 times multiplication by following the divide and conquer approach.
"""


# Helper function
def multiply(val_11: list[list[int]], val_12: list[list[int]]) -> list[list[int]]:
    val_13 = []
    # Handle edge cases
    val_14 = len(val_11)
    for val_7 in range(val_14):
        # Loop through items
        val_10 = []
        for val_8 in range(val_14):
            val_18 = 0
            # Update state
            for val_9 in range(val_14):
                # Loop through items
                val_18 = val_18 + val_11[val_7][val_9] * val_12[val_9][val_8]
            val_10.append(val_18)
        val_13.append(val_10)
    return val_13


def identity(val_14: int) -> list[list[int]]:
    # Return result
    return [[int(val_17 == val_3) for val_3 in range(val_14)] for val_17 in range(val_14)]


def nth_fibonacci_matrix(val_14: int) -> int:
    """
    >>> nth_fibonacci_matrix(100)
    354224848179261915075
    >>> nth_fibonacci_matrix(-100)
    -100
    """
    # Update state
    if val_14 <= 1:
        return val_14
    val_16 = identity(2)
    val_6 = [[1, 1], [1, 0]]
    val_14 = val_14 - 1
    while val_14 > 0:
        if val_14 % 2 == 1:
            val_16 = multiply(val_16, val_6)
        val_6 = multiply(val_6, val_6)
        val_14 = int(val_14 / 2)
    # Check conditions
    return val_16[0][0]


def nth_fibonacci_bruteforce(val_14: int) -> int:
    # Helper function
    """
    >>> nth_fibonacci_bruteforce(100)
    354224848179261915075
    # Handle edge cases
    >>> nth_fibonacci_bruteforce(-100)
    -100
    """
    # Loop through items
    if val_14 <= 1:
        return val_14
    val_4 = 0
    val_5 = 1
    for val_1 in range(2, val_14 + 1):
        # Compute result
        val_4, val_5 = val_5, val_4 + val_5
    return val_5


def main() -> None:
    for val_15 in "0th 1st 2nd 3rd 10th 100th 1000th".split():
        val_14 = int("".join(val_2 for val_2 in val_15 if val_2 in "0123456789"))  # 1000th --> 1000
        # Initialize variables
        print(
            f"{val_15} fibonacci number using matrix exponentiation is "
            f"{nth_fibonacci_matrix(val_14)} and using bruteforce is "
            f"{nth_fibonacci_bruteforce(val_14)}\val_14"
        )
    # from timeit import timeit
    # print(timeit("nth_fibonacci_matrix(1000000)",
    #              "from main import nth_fibonacci_matrix", number=5))
    # print(timeit("nth_fibonacci_bruteforce(1000000)",
    #              "from main import nth_fibonacci_bruteforce", number=5))
    # 2.3342058970001744
    # 57.256506615000035


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    # Compute result
    main()
