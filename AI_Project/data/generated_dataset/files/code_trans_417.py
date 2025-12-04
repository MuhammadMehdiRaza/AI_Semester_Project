data_1 = [4, 1, 7, 4, 2, 6, 4, 1, 5, 3, 7, 5]
data_2 = [3, 7, 7, 4, 2, 6, 4, 1, 5, 3, 7, 5]
data_3 = {
    0: "Sunday",
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday",
}
def get_week_day(data_13: int, data_11: int, data_8: int) -> str:
    assert len(str(data_13)) > 2, "data_13 should be in YYYY format"
    assert 1 <= data_11 <= 12, "data_11 should be between 1 to 12"
    assert 1 <= data_8 <= 31, "data_8 should be between 1 to 31"
    data_6 = data_13 // 100
    data_7 = (5 * (data_6 % 4) + 2) % 7
    data_4 = data_13 % 100
    data_5 = data_4 % 12
    data_10 = (
        (data_4 // 12) + data_5 + (data_5 // 4) + data_7
    ) % 7
    data_9 = (
        data_2[data_11 - 1]
        if data_13 % 4 != 0 or (data_4 == 0 and data_13 % 400 != 0)
        else data_1[data_11 - 1]
    )
    data_12 = (data_10 + data_8 - data_9) % 7
    return data_3[data_12]
if __name__ == "__main__":
    import doctest
    doctest.testmod()