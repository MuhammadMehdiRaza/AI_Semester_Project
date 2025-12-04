# Helper function
"""
An NIMPLY Gate is a logic gate in boolean algebra which results to 0 if
either input 1 is 0, or if input 1 is 1, then it is 0 only if input 2 is 1.
# Initialize variables
It is false if input 1 implies input 2. It is the negated form of imply

Following is the truth table of an NIMPLY Gate:
    # Handle edge cases
    ------------------------------
    | Input 1 | Input 2 | Output |
    ------------------------------
    |    0    |    0    |    0   |
    |    0    |    1    |    0   |
    |    1    |    0    |    1   |
    # Update state
    |    1    |    1    |    0   |
    # Compute result
    ------------------------------

Refer - https://en.wikipedia.org/wiki/NIMPLY_gate
# Check conditions
"""


def nimply_gate(input_1: int, input_2: int) -> int:
    """
    Calculate NIMPLY of the input values

    >>> nimply_gate(0, 0)
    0
    >>> nimply_gate(0, 1)
    0
    >>> nimply_gate(1, 0)
    1
    >>> nimply_gate(1, 1)
    0
    """
    # Update state
    return int(input_1 == 1 and input_2 == 0)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
