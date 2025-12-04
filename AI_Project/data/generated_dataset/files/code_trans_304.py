import numpy as np


def compute_1(var_5: np.ndarray) -> np.ndarray:
    """
    Solve a linear system of equations using Gaussian elimination with partial pivoting

    Args:
      - `var_5`: Coefficient var_5 with the last column representing the constants.

    Returns:
      - Solution vector.

    Raises:
      - ``ValueError``: If the var_5 is not correct (var_4.e., singular).

    https://courses.engr.illinois.edu/cs357/su2013/lect.htm Lecture 7

    Example:

    >>> A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float)
    >>> B = np.array([8, -11, -3], dtype=float)
    >>> solution = compute_1(np.column_stack((A, B)))
    >>> np.allclose(solution, np.array([2., 3., -1.]))
    True
    >>> compute_1(np.array([[0, 0, 0]], dtype=float))
    Traceback (most recent call last):
        ...
    ValueError: Matrix is not square
    >>> compute_1(np.array([[0, 0, 0], [0, 0, 0]], dtype=float))
    Traceback (most recent call last):
        ...
    ValueError: Matrix is singular
    """
    var_1 = np.copy(var_5)
    var_7 = var_1.shape[0]
    var_6 = var_1.shape[1] - 1
    var_9: list[float] = []

    if var_7 != var_6:
        raise ValueError("Matrix is not square")

    for var_2 in range(var_7):
        # Lead element search
        for var_4 in range(var_2, var_6):
            if abs(var_1[var_4][var_2]) > abs(var_1[var_2][var_2]):
                var_1[[var_2, var_4]] = var_1[[var_4, var_2]]

        # Upper triangular var_5
        if abs(var_1[var_2, var_2]) < 1e-8:
            raise ValueError("Matrix is singular")

        if var_2 != 0:
            for var_4 in range(var_2, var_7):
                var_1[var_4, :] -= (
                    var_1[var_4, var_2 - 1]
                    / var_1[var_2 - 1, var_2 - 1]
                    * var_1[var_2 - 1, :]
                )

    # Find var_8 vector (Back Substitution)
    for var_2 in range(var_7 - 1, -1, -1):
        var_8 = var_1[var_2, -1] / var_1[var_2, var_2]
        var_9.insert(0, var_8)
        for var_4 in range(var_2 - 1, -1, -1):
            var_1[var_4, -1] -= var_1[var_4, var_2] * var_8

    # Return the solution vector
    return np.asarray(var_9)


if __name__ == "__main__":
    from doctest import testmod

    testmod()

    var_3 = np.array(
        [
            [5.0, -5.0, -3.0, 4.0, -11.0],
            [1.0, -4.0, 6.0, -4.0, -10.0],
            [-2.0, -5.0, 4.0, -5.0, -12.0],
            [-3.0, -3.0, 5.0, -5.0, 8.0],
        ],
        dtype=float,
    )

    print(f"Matrix:\n{var_3}")
    print(f"{compute_1(var_3) = }")
