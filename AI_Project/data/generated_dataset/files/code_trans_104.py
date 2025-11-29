import string
def compute_1(
    val_3: str, val_6: list[str], val_4: str, val_9: set[str]
) -> list[str]:
    if val_3 == val_4:
        return val_6
    for val_5 in range(len(val_3)):
        for val_2 in string.ascii_lowercase:
            val_8 = val_3[:val_5] + val_2 + val_3[val_5 + 1 :]
            if val_8 in val_9:
                val_9.remove(val_8)
                val_7 = compute_1(
                    val_8, [*val_6, val_8], val_4, val_9
                )
                if val_7:
                    return val_7
                val_9.add(val_8)
    return []
def compute_2(val_1: str, val_4: str, val_9: set[str]) -> list[str]:
    if val_4 not in val_9:
        return []
    return compute_1(val_1, [val_1], val_4, val_9)