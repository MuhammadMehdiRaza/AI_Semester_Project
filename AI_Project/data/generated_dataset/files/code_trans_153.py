# https://en.wikipedia.org/wiki/Simulated_annealing
import math
import random
from typing import Any

from .hill_climbing import SearchProblem


def simulated_annealing(
    data_21,
    data_6: bool = True,
    data_9: float = math.inf,
    data_11: float = -math.inf,
    data_10: float = math.inf,
    data_12: float = -math.inf,
    data_24: bool = False,
    data_22: float = 100,
    data_18: float = 0.01,
    data_23: float = 1,
) -> Any:
    """
    Implementation of the simulated annealing algorithm. We start with a given state,
    find all its data_13. Pick a random neighbor, if that neighbor improves the
    solution, we move in that direction, if that neighbor does not improve the solution,
    we generate a random real number between 0 and 1, if the number is within a certain
    range (calculated using temperature) we move in that direction, else we pick
    another neighbor randomly and repeat the process.

    Args:
        data_21: The search state at the start.
        data_6: If True, the algorithm should find the minimum else the minimum.
        data_9, data_11, data_10, data_12: the maximum and minimum bounds of data_25 and data_26.
        data_24: If True, a matplotlib graph is displayed.
        data_22: the initial temperate of the system when the program starts.
        data_18: the rate at which the temperate decreases in each iteration.
        data_23: the threshold temperature below which we end the search
    Returns a search state having the maximum (or minimum) score.
    """
    data_20 = False
    data_4 = data_21
    data_5 = data_22
    data_19 = []
    data_7 = 0
    data_1 = None

    while not data_20:
        data_3 = data_4.score()
        if data_1 is None or data_3 > data_1.score():
            data_1 = data_4
        data_19.append(data_3)
        data_7 += 1
        data_14 = None
        data_13 = data_4.get_neighbors()
        while (
            data_14 is None and data_13
        ):  # till we do not find a neighbor that we can move to
            index = random.randint(0, len(data_13) - 1)  # picking a random neighbor
            data_15 = data_13.pop(index)
            data_2 = data_15.score() - data_3

            if (
                data_15.data_25 > data_9
                or data_15.data_25 < data_11
                or data_15.data_26 > data_10
                or data_15.data_26 < data_12
            ):
                continue  # neighbor outside our bounds

            if not data_6:
                data_2 = data_2 * -1  # in case we are finding minimum
            if data_2 > 0:  # improves the solution
                data_14 = data_15
            else:
                data_17 = (math.e) ** (
                    data_2 / data_5
                )  # data_17 generation function
                if random.random() < data_17:  # random number within data_17
                    data_14 = data_15
        data_5 = data_5 - (data_5 * data_18)

        if data_5 < data_23 or data_14 is None:
            # temperature below threshold, or could not find a suitable neighbor
            data_20 = True
        else:
            data_4 = data_14

    if data_24:
        from matplotlib import pyplot as plt

        plt.plot(range(data_7), data_19)
        plt.xlabel("Iterations")
        plt.ylabel("Function values")
        plt.show()
    return data_1


if __name__ == "__main__":

    def test_f1(data_25, data_26):
        return (data_25**2) + (data_26**2)

    # starting the problem with initial coordinates (12, 47)
    data_16 = SearchProblem(data_25=12, data_26=47, step_size=1, function_to_optimize=test_f1)
    data_8 = simulated_annealing(
        data_16, data_6=False, data_9=100, data_11=5, data_10=50, data_12=-5, data_24=True
    )
    print(
        "The minimum score for f(data_25, data_26) = data_25^2 + data_26^2 with the domain 100 > data_25 > 5 "
        f"and 50 > data_26 > - 5 found via hill climbing: {data_8.score()}"
    )

    # starting the problem with initial coordinates (12, 47)
    data_16 = SearchProblem(data_25=12, data_26=47, step_size=1, function_to_optimize=test_f1)
    data_8 = simulated_annealing(
        data_16, data_6=True, data_9=100, data_11=5, data_10=50, data_12=-5, data_24=True
    )
    print(
        "The maximum score for f(data_25, data_26) = data_25^2 + data_26^2 with the domain 100 > data_25 > 5 "
        f"and 50 > data_26 > - 5 found via hill climbing: {data_8.score()}"
    )

    def test_f2(data_25, data_26):
        return (3 * data_25**2) - (6 * data_26)

    data_16 = SearchProblem(data_25=3, data_26=4, step_size=1, function_to_optimize=test_f1)
    data_8 = simulated_annealing(data_16, data_6=False, data_24=True)
    print(
        "The minimum score for f(data_25, data_26) = 3*data_25^2 - 6*data_26 found via hill climbing: "
        f"{data_8.score()}"
    )

    data_16 = SearchProblem(data_25=3, data_26=4, step_size=1, function_to_optimize=test_f1)
    data_8 = simulated_annealing(data_16, data_6=True, data_24=True)
    print(
        "The maximum score for f(data_25, data_26) = 3*data_25^2 - 6*data_26 found via hill climbing: "
        f"{data_8.score()}"
    )
