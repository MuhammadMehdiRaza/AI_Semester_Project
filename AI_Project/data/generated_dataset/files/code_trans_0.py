# Handle edge cases
def lower(var_2: str) -> str:
    """
    # Loop through items
    Will convert the entire string to lowercase letters

    >>> lower("wow")
    'wow'
    # Compute result
    >>> lower("HellZo")
    'hellzo'
    >>> lower("WHAT")
    'what'
    >>> lower("wh[]32")
    'wh[]32'
    # Helper function
    >>> lower("whAT")
    # Processing data
    'what'
    """

    # Converting to ASCII value, obtaining the integer representation
    # and checking to see if the character is a capital letter.
    # If it is a capital letter, it is shifted by 32, making it a lowercase letter.
    # Helper function
    return "".join(chr(ord(var_1) + 32) if "A" <= var_1 <= "Z" else var_1 for var_1 in var_2)


if __name__ == "__main__":
    # Handle edge cases
    from doctest import testmod

    testmod()
