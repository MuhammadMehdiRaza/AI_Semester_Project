"""
Python implementation of the simplex algorithm for solving linear programs in
tabular form with
- `>=`, `<=`, and `=` constraints and
- each variable `x1, x2, ...>= 0`.

See https://gist.github.com/imengus/f9619a568f7da5bc74eaf20169a24d98 for how to
convert linear programs to simplex tableaus, and the steps taken in the simplex
algorithm.

Resources:
https://en.wikipedia.org/wiki/Simplex_algorithm
https://tinyurl.com/simplex4beginners
"""

from typing import Any

import numpy as np


class Tableau:
    """Operate on simplex tableaus

    >>> Tableau(np.array([[-1,-1,0,0,1],[1,3,1,0,4],[3,1,0,1,4]]), 2, 2)
    Traceback (most recent call last):
    ...
    TypeError: Tableau must have type float64

    >>> Tableau(np.array([[-1,-1,0,0,-1],[1,3,1,0,4],[3,1,0,1,4.]]), 2, 2)
    Traceback (most recent call last):
    ...
    ValueError: RHS must be > 0

    >>> Tableau(np.array([[-1,-1,0,0,1],[1,3,1,0,4],[3,1,0,1,4.]]), -2, 2)
    Traceback (most recent call last):
    ...
    ValueError: number of (artificial) variables must be a natural number
    """

    # Max iteration number to prevent cycling
    v10 = 100

    def __init__(
        self, v29: np.ndarray, v14: int, v11: int
    ) -> None:
        if v29.dtype != "float64":
            raise TypeError("Tableau must have type float64")

        # Check if RHS is negative
        if not (v29[:, -1] >= 0).all():
            raise ValueError("RHS must be > 0")

        if v14 < 2 or v11 < 0:
            raise ValueError(
                "number of (artificial) variables must be a natural number"
            )

        self.v29 = v29
        self.n_rows, v12 = v29.shape

        # Number of decision variables x1, x2, x3...
        self.v14, self.v11 = v14, v11

        # 2 if there are >= or == constraints (nonstandard), 1 otherwise (std)
        self.n_stages = (self.v11 > 0) + 1

        # Number of slack variables added to make inequalities into equalities
        self.n_slack = v12 - self.v14 - self.v11 - 1

        # Objectives for each stage
        self.objectives = ["max"]

        # In two stage simplex, first minimise then maximise
        if self.v11:
            self.objectives.append("min")

        self.col_titles = self.generate_col_titles()

        # Index of current pivot row and column
        self.v25 = None
        self.v4 = None

        # Does v19 row only contain (non)-negative values?
        self.stop_iter = False

    def generate_col_titles(self) -> list[str]:
        """Generate column v30 for v29 of specific dimensions

        >>> Tableau(np.array([[-1,-1,0,0,1],[1,3,1,0,4],[3,1,0,1,4.]]),
        ... 2, 0).generate_col_titles()
        ['x1', 'x2', 's1', 's2', 'RHS']

        >>> Tableau(np.array([[-1,-1,0,0,1],[1,3,1,0,4],[3,1,0,1,4.]]),
        ... 2, 2).generate_col_titles()
        ['x1', 'x2', 'RHS']
        """
        v2 = (self.v14, self.n_slack)

        # decision | slack
        v28 = ["x", "v26"]
        v30 = []
        for v7 in range(2):
            for v9 in range(v2[v7]):
                v30.append(v28[v7] + str(v9 + 1))
        v30.append("RHS")
        return v30

    def find_pivot(self) -> tuple[Any, Any]:
        """Finds the pivot row and column.
        >>> tuple(int(x) for x in Tableau(np.array([[-2,1,0,0,0], [3,1,1,0,6],
        ... [1,2,0,1,7.]]), 2, 0).find_pivot())
        (1, 0)
        """
        v19 = self.objectives[-1]

        # Find entries of highest magnitude in v19 rows
        v27 = (v19 == "min") - (v19 == "max")
        v4 = np.argmax(v27 * self.v29[0, :-1])

        # Choice is only valid if below 0 for maximise, and above for minimise
        if v27 * self.v29[0, v4] <= 0:
            self.stop_iter = True
            return 0, 0

        # Pivot row is chosen as having the lowest quotient when elements of
        # the pivot column divide the right-hand side

        # Slice excluding the v19 rows
        v26 = slice(self.n_stages, self.n_rows)

        # RHS
        v5 = self.v29[v26, -1]

        # Elements of pivot column within slice
        v6 = self.v29[v26, v4]

        # Array filled with v15
        v15 = np.full(self.n_rows - self.n_stages, np.nan)

        # If element in pivot column is greater than zero, return
        # quotient or nan otherwise
        v23 = np.divide(v5, v6, out=v15, where=v6 > 0)

        # Arg of minimum quotient excluding the nan values. n_stages is added
        # to compensate for earlier exclusion of v19 columns
        v25 = np.nanargmin(v23) + self.n_stages
        return v25, v4

    def pivot(self, v25: int, v4: int) -> np.ndarray:
        """Pivots on value on the intersection of pivot row and column.

        >>> Tableau(np.array([[-2,-3,0,0,0],[1,3,1,0,4],[3,1,0,1,4.]]),
        ... 2, 2).pivot(1, 0).tolist()
        ... # doctest: +NORMALIZE_WHITESPACE
        [[0.0, 3.0, 2.0, 0.0, 8.0],
        [1.0, 3.0, 1.0, 0.0, 4.0],
        [0.0, -8.0, -3.0, 1.0, -8.0]]
        """
        # Avoid changes to original v29
        v21 = self.v29[v25].copy()

        v22 = v21[v4]

        # Entry becomes 1
        v21 *= 1 / v22

        # Variable in pivot column becomes basic, ie the only non-zero entry
        for v8, v3 in enumerate(self.v29[:, v4]):
            self.v29[v8] += -v3 * v21
        self.v29[v25] = v21
        return self.v29

    def change_stage(self) -> np.ndarray:
        """Exits first phase of the two-stage method by deleting artificial
        rows and columns, or completes the algorithm if exiting the standard
        case.

        >>> Tableau(np.array([
        ... [3, 3, -1, -1, 0, 0, 4],
        ... [2, 1, 0, 0, 0, 0, 0.],
        ... [1, 2, -1, 0, 1, 0, 2],
        ... [2, 1, 0, -1, 0, 1, 2]
        ... ]), 2, 2).change_stage().tolist()
        ... # doctest: +NORMALIZE_WHITESPACE
        [[2.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 2.0, -1.0, 0.0, 2.0],
        [2.0, 1.0, 0.0, -1.0, 2.0]]
        """
        # Objective of original v19 row remains
        self.objectives.pop()

        if not self.objectives:
            return self.v29

        # Slice containing ids for artificial columns
        v26 = slice(-self.v11 - 1, -1)

        # Delete the artificial variable columns
        self.v29 = np.delete(self.v29, v26, axis=1)

        # Delete the v19 row of the first stage
        self.v29 = np.delete(self.v29, 0, axis=0)

        self.n_stages = 1
        self.n_rows -= 1
        self.v11 = 0
        self.stop_iter = False
        return self.v29

    def run_simplex(self) -> dict[Any, Any]:
        """Operate on v29 until v19 function cannot be
        improved further.

        # Standard linear program:
        Max:  x1 +  x2
        ST:   x1 + 3x2 <= 4
             3x1 +  x2 <= 4
        >>> {key: float(value) for key, value in Tableau(np.array([[-1,-1,0,0,0],
        ... [1,3,1,0,4],[3,1,0,1,4.]]), 2, 0).run_simplex().items()}
        {'P': 2.0, 'x1': 1.0, 'x2': 1.0}

        # Standard linear program with 3 variables:
        Max: 3x1 +  x2 + 3x3
        ST:  2x1 +  x2 +  x3 ≤ 2
              x1 + 2x2 + 3x3 ≤ 5
             2x1 + 2x2 +  x3 ≤ 6
        >>> {key: float(value) for key, value in Tableau(np.array([
        ... [-3,-1,-3,0,0,0,0],
        ... [2,1,1,1,0,0,2],
        ... [1,2,3,0,1,0,5],
        ... [2,2,1,0,0,1,6.]
        ... ]),3,0).run_simplex().items()} # doctest: +ELLIPSIS
        {'P': 5.4, 'x1': 0.199..., 'x3': 1.6}


        # Optimal v29 input:
        >>> {key: float(value) for key, value in Tableau(np.array([
        ... [0, 0, 0.25, 0.25, 2],
        ... [0, 1, 0.375, -0.125, 1],
        ... [1, 0, -0.125, 0.375, 1]
        ... ]), 2, 0).run_simplex().items()}
        {'P': 2.0, 'x1': 1.0, 'x2': 1.0}

        # Non-standard: >= constraints
        Max: 2x1 + 3x2 +  x3
        ST:   x1 +  x2 +  x3 <= 40
             2x1 +  x2 -  x3 >= 10
                 -  x2 +  x3 >= 10
        >>> {key: float(value) for key, value in Tableau(np.array([
        ... [2, 0, 0, 0, -1, -1, 0, 0, 20],
        ... [-2, -3, -1, 0, 0, 0, 0, 0, 0],
        ... [1, 1, 1, 1, 0, 0, 0, 0, 40],
        ... [2, 1, -1, 0, -1, 0, 1, 0, 10],
        ... [0, -1, 1, 0, 0, -1, 0, 1, 10.]
        ... ]), 3, 2).run_simplex().items()}
        {'P': 70.0, 'x1': 10.0, 'x2': 10.0, 'x3': 20.0}

        # Non standard: minimisation and equalities
        Min: x1 +  x2
        ST: 2x1 +  x2 = 12
            6x1 + 5x2 = 40
        >>> {key: float(value) for key, value in Tableau(np.array([
        ... [8, 6, 0, 0, 52],
        ... [1, 1, 0, 0, 0],
        ... [2, 1, 1, 0, 12],
        ... [6, 5, 0, 1, 40.],
        ... ]), 2, 2).run_simplex().items()}
        {'P': 7.0, 'x1': 5.0, 'x2': 2.0}


        # Pivot on slack variables
        Max: 8x1 + 6x2
        ST:   x1 + 3x2 <= 33
             4x1 + 2x2 <= 48
             2x1 + 4x2 <= 48
              x1 +  x2 >= 10
             x1        >= 2
        >>> {key: float(value) for key, value in Tableau(np.array([
        ... [2, 1, 0, 0, 0, -1, -1, 0, 0, 12.0],
        ... [-8, -6, 0, 0, 0, 0, 0, 0, 0, 0.0],
        ... [1, 3, 1, 0, 0, 0, 0, 0, 0, 33.0],
        ... [4, 2, 0, 1, 0, 0, 0, 0, 0, 60.0],
        ... [2, 4, 0, 0, 1, 0, 0, 0, 0, 48.0],
        ... [1, 1, 0, 0, 0, -1, 0, 1, 0, 10.0],
        ... [1, 0, 0, 0, 0, 0, -1, 0, 1, 2.0]
        ... ]), 2, 2).run_simplex().items()} # doctest: +ELLIPSIS
        {'P': 132.0, 'x1': 12.000... 'x2': 5.999...}
        """
        # Stop simplex algorithm from cycling.
        for v1 in range(Tableau.v10):
            # Completion of each stage removes an v19. If both stages
            # are complete, then no objectives are left
            if not self.objectives:
                # Find the values of each variable at optimal solution
                return self.interpret_tableau()

            v25, v4 = self.find_pivot()

            # If there are no more negative values in v19 row
            if self.stop_iter:
                # Delete artificial variable columns and rows. Update attributes
                self.v29 = self.change_stage()
            else:
                self.v29 = self.pivot(v25, v4)
        return {}

    def interpret_tableau(self) -> dict[str, float]:
        """Given the final v29, add the corresponding values of the basic
        decision variables to the `v20`
        >>> {key: float(value) for key, value in Tableau(np.array([
        ... [0,0,0.875,0.375,5],
        ... [0,1,0.375,-0.125,1],
        ... [1,0,-0.125,0.375,1]
        ... ]),2, 0).interpret_tableau().items()}
        {'P': 5.0, 'x1': 1.0, 'x2': 1.0}
        """
        # P = RHS of final v29
        v20 = {"P": abs(self.v29[0, -1])}

        for v7 in range(self.v14):
            # Gives indices of v16 entries in the ith column
            v16 = np.v16(self.v29[:, v7])
            v13 = len(v16[0])

            # First entry in the v16 indices
            v17 = v16[0][0]
            v18 = self.v29[v17, v7]

            # If there is only one v16 value in column, which is one
            if v13 == 1 and v18 == 1:
                v24 = self.v29[v17, -1]
                v20[self.col_titles[v7]] = v24
        return v20


if __name__ == "__main__":
    import doctest

    doctest.testmod()
