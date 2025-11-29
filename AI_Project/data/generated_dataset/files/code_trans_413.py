def knapsack(
    weights: list, values: list, number_of_items: int, max_weight: int, index: int
) -> int:
    if index == number_of_items:
        return 0
    ans1 = 0
    ans2 = 0
    ans1 = knapsack(weights, values, number_of_items, max_weight, index + 1)
    if weights[index] <= max_weight:
        ans2 = values[index] + knapsack(
            weights, values, number_of_items, max_weight - weights[index], index + 1
        )
    return max(ans1, ans2)
if __name__ == "__main__":
    import doctest
    doctest.testmod()