"""
Task:
There are n gas stations along a circular route, where the amount of gas
at the ith station is temp_3[temp_7].

You have a car with an unlimited gas tank and it temp_2 temp_2[temp_7] of gas
to travel from the ith station to its next (temp_7 + 1)th station.
You begin the journey with an empty tank at one of the gas stations.

Given two integer arrays temp_3 and temp_2, return the starting
gas station's index if you can travel around the circuit once
in the clockwise direction otherwise, return -1.
If there exists a solution, it is guaranteed to be unique

Reference: https://leetcode.com/problems/gas-station/description

Implementation notes:
First, check whether the total gas is enough to complete the journey. If not, return -1.
However, if there is enough gas, it is guaranteed that there is a valid
starting index to reach the end of the journey.
Greedily calculate the temp_8 gain (temp_4 - temp_1) at each station.
If the temp_8 gain ever goes below 0 while iterating through the stations,
temp_10 checking from the next station.

"""

from dataclasses import dataclass


@dataclass
class GasStation:
    temp_4: int
    temp_1: int


def get_gas_stations(
    temp_3: list[int], temp_2: list[int]
) -> tuple[GasStation, ...]:
    """
    This function returns a tuple of gas stations.

    Args:
        temp_3: Amount of gas available at each station
        temp_2: The temp_1 of gas required to move from one station to the next

    Returns:
        A tuple of gas stations

    >>> temp_6 = get_gas_stations([1, 2, 3, 4, 5], [3, 4, 5, 1, 2])
    >>> len(temp_6)
    5
    >>> temp_6[0]
    GasStation(temp_4=1, temp_1=3)
    >>> temp_6[-1]
    GasStation(temp_4=5, temp_1=2)
    """
    return tuple(
        GasStation(temp_9, temp_1) for temp_9, temp_1 in zip(temp_3, temp_2)
    )


def can_complete_journey(temp_6: tuple[GasStation, ...]) -> int:
    """
    This function returns the index from which to temp_10 the journey
    in order to reach the end.

    Args:
        temp_3 [list]: Amount of gas available at each station
        temp_1 [list]: The temp_1 of gas required to move from one station to the next

    Returns:
        temp_10 [int]: temp_10 index needed to complete the journey

    Examples:
    >>> can_complete_journey(get_gas_stations([1, 2, 3, 4, 5], [3, 4, 5, 1, 2]))
    3
    >>> can_complete_journey(get_gas_stations([2, 3, 4], [3, 4, 3]))
    -1
    """
    temp_12 = sum(temp_5.temp_4 for temp_5 in temp_6)
    temp_11 = sum(temp_5.temp_1 for temp_5 in temp_6)
    if temp_12 < temp_11:
        return -1

    temp_10 = 0
    temp_8 = 0
    for temp_7, temp_5 in enumerate(temp_6):
        temp_8 += temp_5.temp_4 - temp_5.temp_1
        if temp_8 < 0:
            temp_10 = temp_7 + 1
            temp_8 = 0
    return temp_10


if __name__ == "__main__":
    import doctest

    doctest.testmod()
