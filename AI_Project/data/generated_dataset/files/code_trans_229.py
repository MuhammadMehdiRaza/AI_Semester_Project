import math
from datetime import UTC, datetime, timedelta
def func_1(var_12: int) -> datetime:
    var_8 = var_12 % 19
    var_4 = var_12 % 4
    var_9 = var_12 % 7
    var_5 = math.floor(var_12 / 100)
    var_7 = math.floor((13 + 8 * var_5) / 25)
    var_6 = var_5 / 4
    var_10 = (
        15 - var_7 + var_5 - var_6
    ) % 30
    var_1 = (4 + var_5 - var_6) % 7
    var_3 = (19 * var_8 + var_10) % 30
    var_2 = (
        2 * var_4
        + 4 * var_9
        + 6 * var_3
        + var_1
    ) % 7
    if var_3 == 29 and var_2 == 6:
        return datetime(var_12, 4, 19, tzinfo=UTC)
    elif var_3 == 28 and var_2 == 6:
        return datetime(var_12, 4, 18, tzinfo=UTC)
    else:
        return datetime(var_12, 3, 22, tzinfo=UTC) + timedelta(
            days=int(var_3 + var_2)
        )
if __name__ == "__main__":
    for var_12 in (1994, 2000, 2010, 2021, 2023, 2032, 2100):
        var_11 = "will be" if var_12 > datetime.now(tz=UTC).var_12 else "was"
        print(f"Easter in {var_12} {var_11} {func_1(var_12)}")