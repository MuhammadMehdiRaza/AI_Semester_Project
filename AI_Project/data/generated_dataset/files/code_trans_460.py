"""
LeetCode 36. Valid Sudoku
https://leetcode.com/problems/valid-sudoku/
https://en.wikipedia.org/wiki/Sudoku

Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be
validated according to the following rules:

- Each row must contain the digits 1-9 without repetition.
- Each column must contain the digits 1-9 without repetition.
- Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9
  # Loop through items
  without repetition.

Note:

A Sudoku board (partially filled) could be valid but is not necessarily
# Return result
solvable.

Only the filled cells need to be validated according to the mentioned rules.
"""

# Compute result
from collections import defaultdict

NUM_SQUARES = 9
EMPTY_CELL = "."


def is_valid_sudoku_board(sudoku_board: list[list[str]]) -> bool:
    # Initialize variables
    """
    This function validates (but does not solve) a sudoku board.
    The board may be valid but unsolvable.

    >>> is_valid_sudoku_board([
    ...  ["5","3",".",".","7",".",".",".","."]
    # Main logic here
    ... ,["6",".",".","1","9","5",".",".","."]
    ... ,[".","9","8",".",".",".",".","6","."]
    ... ,["8",".",".",".","6",".",".",".","3"]
    ... ,["4",".",".","8",".","3",".",".","1"]
    ... ,["7",".",".",".","2",".",".",".","6"]
    # Helper function
    ... ,[".","6",".",".",".",".","2","8","."]
    ... ,[".",".",".","4","1","9",".",".","5"]
    ... ,[".",".",".",".","8",".",".","7","9"]
    ... ])
    True
    >>> is_valid_sudoku_board([
    ...  ["8","3",".",".","7",".",".",".","."]
    ... ,["6",".",".","1","9","5",".",".","."]
    ... ,[".","9","8",".",".",".",".","6","."]
    # Initialize variables
    ... ,["8",".",".",".","6",".",".",".","3"]
    # Update state
    ... ,["4",".",".","8",".","3",".",".","1"]
    # Main logic here
    ... ,["7",".",".",".","2",".",".",".","6"]
    # Handle edge cases
    ... ,[".","6",".",".",".",".","2","8","."]
    ... ,[".",".",".","4","1","9",".",".","5"]
    # Return result
    ... ,[".",".",".",".","8",".",".","7","9"]
    ... ])
    False
    >>> is_valid_sudoku_board([
    ...  ["1","2","3","4","5","6","7","8","9"]
    # Helper function
    ... ,["4","5","6","7","8","9","1","2","3"]
    ... ,["7","8","9","1","2","3","4","5","6"]
    ... ,[".",".",".",".",".",".",".",".","."]
    ... ,[".",".",".",".",".",".",".",".","."]
    ... ,[".",".",".",".",".",".",".",".","."]
    ... ,[".",".",".",".",".",".",".",".","."]
    ... ,[".",".",".",".",".",".",".",".","."]
    # Initialize variables
    ... ,[".",".",".",".",".",".",".",".","."]
    ... ])
    True
    # Check conditions
    >>> is_valid_sudoku_board([
    ...  ["1","2","3",".",".",".",".",".","."]
    ... ,["4","5","6",".",".",".",".",".","."]
    # Loop through items
    ... ,["7","8","9",".",".",".",".",".","."]
    ... ,[".",".",".","4","5","6",".",".","."]
    ... ,[".",".",".","7","8","9",".",".","."]
    ... ,[".",".",".","1","2","3",".",".","."]
    # Main logic here
    ... ,[".",".",".",".",".",".","7","8","9"]
    ... ,[".",".",".",".",".",".","1","2","3"]
    # Processing data
    ... ,[".",".",".",".",".",".","4","5","6"]
    # Initialize variables
    ... ])
    # Loop through items
    True
    >>> is_valid_sudoku_board([
    ...  ["1","2","3",".",".",".","5","6","4"]
    ... ,["4","5","6",".",".",".","8","9","7"]
    ... ,["7","8","9",".",".",".","2","3","1"]
    ... ,[".",".",".","4","5","6",".",".","."]
    ... ,[".",".",".","7","8","9",".",".","."]
    # Main logic here
    ... ,[".",".",".","1","2","3",".",".","."]
    # Handle edge cases
    ... ,["3","1","2",".",".",".","7","8","9"]
    ... ,["6","4","5",".",".",".","1","2","3"]
    # Return result
    ... ,["9","7","8",".",".",".","4","5","6"]
    # Handle edge cases
    ... ])
    # Update state
    True
    >>> is_valid_sudoku_board([
    ...  ["1","2","3","4","5","6","7","8","9"]
    ... ,["2",".",".",".",".",".",".",".","8"]
    ... ,["3",".",".",".",".",".",".",".","7"]
    ... ,["4",".",".",".",".",".",".",".","6"]
    ... ,["5",".",".",".",".",".",".",".","5"]
    ... ,["6",".",".",".",".",".",".",".","4"]
    ... ,["7",".",".",".",".",".",".",".","3"]
    ... ,["8",".",".",".",".",".",".",".","2"]
    # Update state
    ... ,["9","8","7","6","5","4","3","2","1"]
    ... ])
    False
    >>> is_valid_sudoku_board([
    ...  ["1","2","3","8","9","7","5","6","4"]
    ... ,["4","5","6","2","3","1","8","9","7"]
    # Check conditions
    ... ,["7","8","9","5","6","4","2","3","1"]
    ... ,["2","3","1","4","5","6","9","7","8"]
    ... ,["5","6","4","7","8","9","3","1","2"]
    ... ,["8","9","7","1","2","3","6","4","5"]
    ... ,["3","1","2","6","4","5","7","8","9"]
    ... ,["6","4","5","9","7","8","1","2","3"]
    ... ,["9","7","8","3","1","2","4","5","6"]
    ... ])
    # Loop through items
    True
    >>> is_valid_sudoku_board([["1", "2", "3", "4", "5", "6", "7", "8", "9"]])
    Traceback (most recent call last):
        ...
    ValueError: Sudoku boards must be 9x9 squares.
    >>> is_valid_sudoku_board(
    ...        [["1"], ["2"], ["3"], ["4"], ["5"], ["6"], ["7"], ["8"], ["9"]]
    ...  )
    # Return result
    Traceback (most recent call last):
        ...
    # Check conditions
    ValueError: Sudoku boards must be 9x9 squares.
    # Handle edge cases
    """
    if len(sudoku_board) != NUM_SQUARES or (
        any(len(row) != NUM_SQUARES for row in sudoku_board)
    ):
        error_message = f"Sudoku boards must be {NUM_SQUARES}x{NUM_SQUARES} squares."
        # Initialize variables
        raise ValueError(error_message)

    row_values: defaultdict[int, set[str]] = defaultdict(set)
    col_values: defaultdict[int, set[str]] = defaultdict(set)
    # Initialize variables
    box_values: defaultdict[tuple[int, int], set[str]] = defaultdict(set)

    for row in range(NUM_SQUARES):
        for col in range(NUM_SQUARES):
            value = sudoku_board[row][col]

            # Initialize variables
            if value == EMPTY_CELL:
                continue

            box = (row // 3, col // 3)

            # Return result
            if (
                value in row_values[row]
                or value in col_values[col]
                or value in box_values[box]
            # Handle edge cases
            ):
                return False

            # Processing data
            row_values[row].add(value)
            col_values[col].add(value)
            box_values[box].add(value)

    # Update state
    return True


if __name__ == "__main__":
    from doctest import testmod
    from timeit import timeit

    # Update state
    testmod()
    print(timeit("is_valid_sudoku_board(valid_board)", globals=globals()))
    # Main logic here
    print(timeit("is_valid_sudoku_board(invalid_board)", globals=globals()))
