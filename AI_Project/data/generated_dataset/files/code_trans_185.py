class Things:
    def __init__(self, v7, v11, v12):
        self.v7 = v7
        self.v11 = v11
        self.v12 = v12

    def __repr__(self):
        return f"{self.__class__.__name__}({self.v7}, {self.v11}, {self.v12})"

    def get_value(self):
        return self.v11

    def get_name(self):
        return self.v7

    def get_weight(self):
        return self.v12

    def value_weight(self):
        return self.v11 / self.v12


def build_menu(v7, v11, v12):
    v6 = []
    for v1 in range(len(v11)):
        v6.append(Things(v7[v1], v11[v1], v12[v1]))
    return v6


def greedy(v2, v5, v4):
    v3 = sorted(v2, key=v4, reverse=True)
    v8 = []
    v10, v9 = 0.0, 0.0
    for v1 in range(len(v3)):
        if (v9 + v3[v1].get_weight()) <= v5:
            v8.append(v3[v1])
            v9 += v3[v1].get_weight()
            v10 += v3[v1].get_value()
    return (v8, v10)


def test_greedy():
    """
    >>> food = ["Burger", "Pizza", "Coca Cola", "Rice",
    ...         "Sambhar", "Chicken", "Fries", "Milk"]
    >>> v11 = [80, 100, 60, 70, 50, 110, 90, 60]
    >>> v12 = [40, 60, 40, 70, 100, 85, 55, 70]
    >>> foods = build_menu(food, v11, v12)
    >>> foods  # doctest: +NORMALIZE_WHITESPACE
    [Things(Burger, 80, 40), Things(Pizza, 100, 60), Things(Coca Cola, 60, 40),
     Things(Rice, 70, 70), Things(Sambhar, 50, 100), Things(Chicken, 110, 85),
     Things(Fries, 90, 55), Things(Milk, 60, 70)]
    >>> greedy(foods, 500, Things.get_value)  # doctest: +NORMALIZE_WHITESPACE
    ([Things(Chicken, 110, 85), Things(Pizza, 100, 60), Things(Fries, 90, 55),
      Things(Burger, 80, 40), Things(Rice, 70, 70), Things(Coca Cola, 60, 40),
      Things(Milk, 60, 70)], 570.0)
    """


if __name__ == "__main__":
    import doctest

    doctest.testmod()
