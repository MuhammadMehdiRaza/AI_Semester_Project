from __future__ import annotations

v1 = [
    [-1, 0],  # left
    [0, -1],  # down
    [1, 0],  # right
    [0, 1],  # up
]


# function to search the v20
def search(
    v13: list[list[int]],
    v16: list[int],
    v12: list[int],
    v6: int,
    v14: list[list[int]],
) -> tuple[list[list[int]], list[list[int]]]:
    """
    Search for a v20 on a v13 avoiding obstacles.
    >>> v13 = [[0, 1, 0, 0, 0, 0],
    ...         [0, 1, 0, 0, 0, 0],
    ...         [0, 1, 0, 0, 0, 0],
    ...         [0, 1, 0, 0, 1, 0],
    ...         [0, 0, 0, 0, 1, 0]]
    >>> v16 = [0, 0]
    >>> v12 = [len(v13) - 1, len(v13[0]) - 1]
    >>> v6 = 1
    >>> v14 = [[0] * len(v13[0]) for _ in range(len(v13))]
    >>> v14 = [[0 for v22 in range(len(v13[0]))] for v5 in range(len(v13))]
    >>> for v15 in range(len(v13)):
    ...     for v18 in range(len(v13[0])):
    ...         v14[v15][v18] = abs(v15 - v12[0]) + abs(v18 - v12[1])
    ...         if v13[v15][v18] == 1:
    ...             v14[v15][v18] = 99
    >>> v20, v2 = search(v13, v16, v12, v6, v14)
    >>> v20  # doctest: +NORMALIZE_WHITESPACE
    [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [4, 1], [4, 2], [4, 3], [3, 3],
    [2, 3], [2, 4], [2, 5], [3, 5], [4, 5]]
    >>> v2  # doctest: +NORMALIZE_WHITESPACE
    [[0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0], [2, 0, 0, 0, 3, 3],
    [2, 0, 0, 0, 0, 2], [2, 3, 3, 3, 0, 2]]
    """
    v4 = [
        [0 for v5 in range(len(v13[0]))] for v22 in range(len(v13))
    ]  # the reference v13
    v4[v16[0]][v16[1]] = 1
    v2 = [
        [0 for v5 in range(len(v13[0]))] for v22 in range(len(v13))
    ]  # the v2 v13

    v23 = v16[0]
    v25 = v16[1]
    v10 = 0
    v7 = v10 + v14[v23][v25]  # v6 from starting v3 to destination v3
    v3 = [[v7, v10, v23, v25]]

    v9 = False  # flag that is set when search is complete
    v21 = False  # flag set if we can't find expand

    while not v9 and not v21:
        if len(v3) == 0:
            raise ValueError("Algorithm is unable to find solution")
        else:  # to choose the least costliest v2 so as to move closer to the v12
            v3.sort()
            v3.reverse()
            v19 = v3.pop()
            v23 = v19[2]
            v25 = v19[3]
            v10 = v19[1]

            if v23 == v12[0] and v25 == v12[1]:
                v9 = True
            else:
                for v15 in range(len(v1)):  # to try out different valid actions
                    v24 = v23 + v1[v15][0]
                    v26 = v25 + v1[v15][1]
                    if (
                        v24 >= 0
                        and v24 < len(v13)
                        and v26 >= 0
                        and v26 < len(v13[0])
                        and v4[v24][v26] == 0
                        and v13[v24][v26] == 0
                    ):
                        v11 = v10 + v6
                        v8 = v11 + v14[v24][v26]
                        v3.append([v8, v11, v24, v26])
                        v4[v24][v26] = 1
                        v2[v24][v26] = v15
    v17 = []
    v23 = v12[0]
    v25 = v12[1]
    v17.append([v23, v25])  # we get the reverse v20 from here
    while v23 != v16[0] or v25 != v16[1]:
        v24 = v23 - v1[v2[v23][v25]][0]
        v26 = v25 - v1[v2[v23][v25]][1]
        v23 = v24
        v25 = v26
        v17.append([v23, v25])

    v20 = []
    for v15 in range(len(v17)):
        v20.append(v17[len(v17) - 1 - v15])
    return v20, v2


if __name__ == "__main__":
    v13 = [
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],  # 0 are free v20 whereas 1's are obstacles
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0],
    ]

    v16 = [0, 0]
    # all coordinates are given in format [v25,v23]
    v12 = [len(v13) - 1, len(v13[0]) - 1]
    v6 = 1

    # the v6 map which pushes the v20 closer to the v12
    v14 = [[0 for v22 in range(len(v13[0]))] for v5 in range(len(v13))]
    for v15 in range(len(v13)):
        for v18 in range(len(v13[0])):
            v14[v15][v18] = abs(v15 - v12[0]) + abs(v18 - v12[1])
            if v13[v15][v18] == 1:
                # added extra penalty in the v14 map
                v14[v15][v18] = 99

    v20, v2 = search(v13, v16, v12, v6, v14)

    print("ACTION MAP")
    for v15 in range(len(v2)):
        print(v2[v15])

    for v15 in range(len(v20)):
        print(v20[v15])
