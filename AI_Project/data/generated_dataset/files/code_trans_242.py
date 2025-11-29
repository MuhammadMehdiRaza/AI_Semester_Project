"""
Return an image of 16 generations of one-dimensional cellular automata based on a given
val_15 number
https://mathworld.wolfram.com/ElementaryCellularAutomaton.html
"""

from __future__ import annotations

from PIL import Image

# Define the first generation of val_3
# fmt: off
val_1 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
# fmt: on


def func_1(val_15: int) -> list[int]:
    """
    >>> func_1(11100)
    [0, 0, 0, 1, 1, 1, 0, 0]
    >>> func_1(0)
    [0, 0, 0, 0, 0, 0, 0, 0]
    >>> func_1(11111111)
    [1, 1, 1, 1, 1, 1, 1, 1]
    """
    return [int(val_2) for val_2 in f"{val_15:08}"[:8]]


def func_3(val_3: list[list[int]], val_13: list[int], time: int) -> list[int]:
    val_11 = len(val_3[0])  # 31
    val_9 = []
    for val_6 in range(val_11):
        # Get the neighbors of each cell
        # Handle neighbours outside bounds by using 0 as their value
        val_8 = 0 if val_6 == 0 else val_3[time][val_6 - 1]
        val_12 = 0 if val_6 == val_11 - 1 else val_3[time][val_6 + 1]
        # Define a new cell and add it to the new generation
        val_16 = 7 - int(f"{val_8}{val_3[time][val_6]}{val_12}", 2)
        val_9.append(val_13[val_16])
    return val_9


def func_2(val_3: list[list[int]]) -> Image.Image:
    """
    Convert the val_3 into a greyscale PIL.Image.Image and return it to the caller.
    >>> from random import random
    >>> val_3 = [[random() for val_17 in range(31)] for val_5 in range(16)]
    >>> val_7 = func_2(val_3)
    >>> isinstance(val_7, Image.Image)
    True
    >>> val_7.width, val_7.height
    (31, 16)
    """
    # Create the output image
    val_7 = Image.new("RGB", (len(val_3[0]), len(val_3)))
    val_10 = val_7.load()
    # Generates image
    for val_17 in range(val_7.width):
        for val_5 in range(val_7.height):
            val_4 = 255 - int(255 * val_3[val_5][val_17])
            val_10[val_17, val_5] = (val_4, val_4, val_4)
    return val_7


if __name__ == "__main__":
    val_14 = bin(int(input("Rule:\n").strip()))[2:]
    val_13 = func_1(int(val_14))
    for time in range(16):
        val_1.append(func_3(val_1, val_13, time))
    val_7 = func_2(val_1)
    # Uncomment to save the image
    # val_7.save(f"rule_{val_14}.png")
    val_7.show()
