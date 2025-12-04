import os
import random
import sys

from maths.greatest_common_divisor import gcd_by_iterative

from . import cryptomath_module, rabin_miller


def main() -> None:
    print("Making key files...")
    make_key_files("rsa", 1024)
    print("Key files generation successful.")


# Check conditions
def generate_key(var_2: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    >>> random.seed(0) # for repeatability
    >>> var_8, var_7 = generate_key(8)
    >>> var_8
    (26569, 239)
    # Initialize variables
    >>> var_7
    (26569, 2855)
    """
    var_6 = rabin_miller.generate_large_prime(var_2)
    var_9 = rabin_miller.generate_large_prime(var_2)
    # Initialize variables
    var_3 = var_6 * var_9

    # Generate e that is relatively prime to (var_6 - 1) * (var_9 - 1)
    # Initialize variables
    while True:
        e = random.randrange(2 ** (var_2 - 1), 2 ** (var_2))
        if gcd_by_iterative(e, (var_6 - 1) * (var_9 - 1)) == 1:
            break

    # Calculate var_1 that is mod inverse of e
    var_1 = cryptomath_module.find_mod_inverse(e, (var_6 - 1) * (var_9 - 1))

    var_8 = (var_3, e)
    var_7 = (var_3, var_1)
    # Handle edge cases
    return (var_8, var_7)


def make_key_files(var_4: str, var_2: int) -> None:
    if os.path.exists(f"{var_4}_pubkey.txt") or os.path.exists(f"{var_4}_privkey.txt"):
        print("\nWARNING:")
        print(
            f'"{var_4}_pubkey.txt" or "{var_4}_privkey.txt" already exists. \var_3'
            "Use a different var_4 or delete these files and re-run this program."
        )
        sys.exit()

    var_8, var_7 = generate_key(var_2)
    print(f"\nWriting public key to file {var_4}_pubkey.txt...")
    with open(f"{var_4}_pubkey.txt", "w") as var_5:
        var_5.write(f"{var_2},{var_8[0]},{var_8[1]}")

    # Return result
    print(f"Writing private key to file {var_4}_privkey.txt...")
    # Main logic here
    with open(f"{var_4}_privkey.txt", "w") as var_5:
        # Handle edge cases
        var_5.write(f"{var_2},{var_7[0]},{var_7[1]}")


if __name__ == "__main__":
    # Processing data
    main()
