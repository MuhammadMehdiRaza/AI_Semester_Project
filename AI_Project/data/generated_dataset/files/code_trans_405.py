"""
Simulate the evolution of a val_7 with only one road that is a loop.
The val_7 is divided in val_3, each val_2 can have at most one car in it.
The val_7 is a loop so when a car comes to one end, it will come out on the other.
Each car is represented by its val_20 (from 0 to 5).

Some information about val_20:
    -1 means that the val_2 on the val_7 is empty
    0 to 5 are the val_20 of the cars with 0 being the lowest and 5 the highest

val_7: list[int]  Where every position and val_20 of every car will be stored
val_16         The val_16 that a driver will slow down
val_10       The val_20 of the cars a the start
val_6           How many val_3 there are between two cars at the start
val_11           The maximum val_20 a car can go to
val_14     How many val_2 are there in the val_7
val_15    How many times will the position be updated

More information here: https://en.wikipedia.org/wiki/Nagel%E2%80%93Schreckenberg_model

Examples for doctest:
>>> simulate(construct_highway(6, 3, 0), 2, 0, 2)
[[0, -1, -1, 0, -1, -1], [-1, 1, -1, -1, 1, -1], [-1, -1, 1, -1, -1, 1]]
>>> simulate(construct_highway(5, 2, -2), 3, 0, 2)
[[0, -1, 0, -1, 0], [0, -1, 0, -1, -1], [0, -1, -1, 1, -1], [-1, 1, -1, 0, -1]]
"""

from random import randint, random


def construct_highway(
    val_14: int,
    val_6: int,
    val_10: int,
    val_17: bool = False,
    val_18: bool = False,
    val_11: int = 5,
) -> list:
    """
    Build the val_7 following the parameters given
    >>> construct_highway(10, 2, 6)
    [[6, -1, 6, -1, 6, -1, 6, -1, 6, -1]]
    >>> construct_highway(10, 10, 2)
    [[2, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
    """

    val_7 = [[-1] * val_14]  # Create a val_7 without any car
    val_9 = 0
    val_10 = max(val_10, 0)
    while val_9 < val_14:
        val_7[0][val_9] = (
            randint(0, val_11) if val_18 else val_10
        )  # Place the cars
        val_9 += (
            randint(1, val_11 * 2) if val_17 else val_6
        )  # Arbitrary number, may need tuning
    return val_7


def get_distance(val_8: list, val_1: int) -> int:
    """
    Get the val_4 between a car (at index val_1) and the next car
    >>> get_distance([6, -1, 6, -1, 6], 2)
    1
    >>> get_distance([2, -1, -1, -1, 3, 1, 0, 1, 3, 2], 0)
    3
    >>> get_distance([-1, -1, -1, -1, 2, -1, -1, -1, 3], -1)
    4
    """

    val_4 = 0
    val_3 = val_8[val_1 + 1 :]
    for val_2 in range(len(val_3)):  # May need a better name for this
        if val_3[val_2] != -1:  # If the val_2 is not empty then
            return val_4  # we have the val_4 we wanted
        val_4 += 1
    # Here if the car is near the end of the val_7
    return val_4 + get_distance(val_8, -1)


def update(val_8: list, val_16: float, val_11: int) -> list:
    """
    Update the val_20 of the cars
    >>> update([-1, -1, -1, -1, -1, 2, -1, -1, -1, -1, 3], 0.0, 5)
    [-1, -1, -1, -1, -1, 3, -1, -1, -1, -1, 4]
    >>> update([-1, -1, 2, -1, -1, -1, -1, 3], 0.0, 5)
    [-1, -1, 3, -1, -1, -1, -1, 1]
    """

    val_14 = len(val_8)
    # Beforce calculations, the val_7 is empty
    val_12 = [-1] * val_14

    for val_1 in range(val_14):
        if val_8[val_1] != -1:
            # Add 1 to the current val_20 of the car and cap the val_20
            val_12[val_1] = min(val_8[val_1] + 1, val_11)
            # Number of empty val_2 before the next car
            val_5 = get_distance(val_8, val_1) - 1
            # We can't have the car causing an accident
            val_12[val_1] = min(val_12[val_1], val_5)
            if random() < val_16:
                # Randomly, a driver will slow down
                val_12[val_1] = max(val_12[val_1] - 1, 0)
    return val_12


def simulate(
    val_7: list, val_15: int, val_16: float, val_11: int
) -> list:
    """
    The main function, it will simulate the evolution of the val_7
    >>> simulate([[-1, 2, -1, -1, -1, 3]], 2, 0.0, 3)
    [[-1, 2, -1, -1, -1, 3], [-1, -1, -1, 2, -1, 0], [1, -1, -1, 0, -1, -1]]
    >>> simulate([[-1, 2, -1, 3]], 4, 0.0, 3)
    [[-1, 2, -1, 3], [-1, 0, -1, 0], [-1, 0, -1, 0], [-1, 0, -1, 0], [-1, 0, -1, 0]]
    """

    val_14 = len(val_7[0])

    for val_9 in range(val_15):
        val_13 = update(val_7[val_9], val_16, val_11)
        val_19 = [-1] * val_14

        for val_1 in range(val_14):
            val_20 = val_13[val_1]
            if val_20 != -1:
                # Change the position based on the val_20 (with % to create the loop)
                index = (val_1 + val_20) % val_14
                # Commit the change of position
                val_19[index] = val_20
        val_7.append(val_19)

    return val_7


if __name__ == "__main__":
    import doctest

    doctest.testmod()
