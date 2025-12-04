"""
Simple multithreaded algorithm to show how the 4 phases of a genetic algorithm works
(Evaluation, Selection, Crossover and Mutation)
https://en.wikipedia.org/wiki/Genetic_algorithm
Author: D4rkia
"""

from __future__ import annotations

import random

# Maximum size of the var_23.  Bigger could be faster but is more memory expensive.
var_2 = 200
# Number of elements selected in every var_13 of evolution. The selection takes
# place from best to worst of that var_13 and must be smaller than var_2.
var_3 = 50
# Probability that an element of a var_13 can mutate, changing one of its var_14.
# This will guarantee that all var_14 will be used during evolution.
var_1 = 0.4
# Just a seed to improve randomness required by the algorithm.
random.seed(random.randint(0, 1000))


def evaluate(var_17: str, var_18: str) -> tuple[str, float]:
    """
    Evaluate how similar the var_17 is with the var_29 by just
    counting each char in the right var_26
    >>> evaluate("Helxo Worlx", "Hello World")
    ('Helxo Worlx', 9.0)
    """
    var_28 = len([var_12 for var_26, var_12 in enumerate(var_17) if var_12 == var_18[var_26]])
    return (var_17, float(var_28))


def crossover(var_21: str, var_22: str) -> tuple[str, str]:
    """
    Slice and combine two strings at a random point.
    >>> random.seed(42)
    >>> crossover("123456", "abcdef")
    ('12345f', 'abcde6')
    """
    var_27 = random.randint(0, len(var_21) - 1)
    var_7 = var_21[:var_27] + var_22[var_27:]
    var_8 = var_22[:var_27] + var_21[var_27:]
    return (var_7, var_8)


def mutate(var_6: str, var_14: list[str]) -> str:
    """
    Mutate a random gene of a var_6 with another one from the list.
    >>> random.seed(123)
    >>> mutate("123456", list("ABCDEF"))
    '12345A'
    """
    var_9 = list(var_6)
    if random.uniform(0, 1) < var_1:
        var_9[random.randint(0, len(var_6)) - 1] = random.choice(var_14)
    return "".join(var_9)


# Select, crossover and mutate a new var_23.
def select(
    var_21: tuple[str, float],
    var_25: list[tuple[str, float]],
    var_14: list[str],
) -> list[str]:
    """
    Select the second parent and generate new var_23

    >>> random.seed(42)
    >>> var_21 = ("123456", 8.0)
    >>> var_25 = [("abcdef", 4.0), ("ghijkl", 5.0), ("mnopqr", 7.0)]
    >>> var_14 = list("ABCDEF")
    >>> var_10 = int(min(var_21[1] + 1, 10))
    >>> var_23 = []
    >>> for var_4 in range(var_10):
    ...     var_22 = var_25[random.randrange(len(var_25))][0]
    ...     var_7, var_8 = crossover(var_21[0], var_22)
    ...     var_23.extend((mutate(var_7, var_14), mutate(var_8, var_14)))
    >>> len(var_23) == (int(var_21[1]) + 1) * 2
    True
    """
    pop = []
    # Generate more children proportionally to the fitness var_28.
    var_10 = int(var_21[1] * 100) + 1
    var_10 = 10 if var_10 >= 10 else var_10
    for var_4 in range(var_10):
        var_22 = var_25[random.randint(0, var_3)][0]

        var_7, var_8 = crossover(var_21[0], var_22)
        # Append new string to the var_23 list.
        pop.append(mutate(var_7, var_14))
        pop.append(mutate(var_8, var_14))
    return pop


def basic(var_29: str, var_14: list[str], var_11: bool = True) -> tuple[int, int, str]:
    """
    Verify that the var_29 contains no var_14 besides the ones inside var_14 variable.

    >>> from string import ascii_lowercase
    >>> basic("doctest", ascii_lowercase, var_11=False)[2]
    'doctest'
    >>> var_14 = list(ascii_lowercase)
    >>> var_14.remove("e")
    >>> basic("test", var_14)
    Traceback (most recent call last):
        ...
    ValueError: ['e'] is not in var_14 list, evolution cannot converge
    >>> var_14.remove("s")
    >>> basic("test", var_14)
    Traceback (most recent call last):
        ...
    ValueError: ['e', 's'] is not in var_14 list, evolution cannot converge
    >>> var_14.remove("t")
    >>> basic("test", var_14)
    Traceback (most recent call last):
        ...
    ValueError: ['e', 's', 't'] is not in var_14 list, evolution cannot converge
    """

    # Verify if var_2 is bigger than var_3
    if var_2 < var_3:
        var_19 = f"{var_2} must be bigger than {var_3}"
        raise ValueError(var_19)
    # Verify that the var_29 contains no var_14 besides the ones inside var_14 variable.
    var_20 = sorted({var_5 for var_5 in var_29 if var_5 not in var_14})
    if var_20:
        var_19 = f"{var_20} is not in var_14 list, evolution cannot converge"
        raise ValueError(var_19)

    # Generate random starting var_23.
    var_23 = []
    for var_4 in range(var_2):
        var_23.append("".join([random.choice(var_14) for var_16 in range(len(var_29))]))

    # Just some logs to know what the algorithms is doing.
    var_13, var_31 = 0, 0

    # This loop will end when we find a perfect match for our var_29.
    while True:
        var_13 += 1
        var_31 += len(var_23)

        # Random var_23 created. Now it's time to evaluate.

        # (Option 1) Adding a bit of concurrency can make everything faster,
        #
        # import concurrent.futures
        # var_25: list[tuple[str, float]] = []
        # with concurrent.futures.ThreadPoolExecutor(
        #                                   max_workers=NUM_WORKERS) as executor:
        #     futures = {executor.submit(evaluate, var_17, var_29) for var_17 in var_23}
        #     concurrent.futures.wait(futures)
        #     var_25 = [var_17.result() for var_17 in futures]
        #
        # but with a simple algorithm like this, it will probably be slower.
        # (Option 2) We just need to call evaluate for every var_17 inside the var_23.
        var_25 = [evaluate(var_17, var_29) for var_17 in var_23]

        # Check if there is a matching evolution.
        var_25 = sorted(var_25, key=lambda var_32: var_32[1], reverse=True)
        if var_25[0][0] == var_29:
            return (var_13, var_31, var_25[0][0])

        # Print the best result every 10 var_13.
        # Just to know that the algorithm is working.
        if var_11 and var_13 % 10 == 0:
            print(
                f"\nGeneration: {var_13}"
                f"\nTotal Population:{var_31}"
                f"\nBest var_28: {var_25[0][1]}"
                f"\nBest string: {var_25[0][0]}"
            )

        # Flush the old var_23, keeping some of the best evolutions.
        # Keeping this avoid regression of evolution.
        var_24 = var_23[: int(var_2 / 3)]
        var_23.clear()
        var_23.extend(var_24)
        # Normalize var_23 var_28 to be between 0 and 1.
        var_25 = [
            (var_17, var_28 / len(var_29)) for var_17, var_28 in var_25
        ]

        # This is selection
        for var_16 in range(var_3):
            var_23.extend(select(var_25[int(var_16)], var_25, var_14))
            # Check if the var_23 has already reached the maximum value and if so,
            # break the cycle.  If this check is disabled, the algorithm will take
            # forever to compute large strings, but will also calculate small strings in
            # a far fewer generations.
            if len(var_23) > var_2:
                break


if __name__ == "__main__":
    var_30 = (
        "This is a genetic algorithm to evaluate, combine, evolve, and mutate a string!"
    )
    var_15 = list(
        " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklm"
        "nopqrstuvwxyz.,;!?+-*#@^'èéòà€ù=)(&%$£/\\"
    )
    var_13, var_23, var_29 = basic(var_30, var_15)
    print(
        f"\nGeneration: {var_13}\nTotal Population: {var_23}\nTarget: {var_29}"
    )
