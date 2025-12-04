#!/usr/bin/env python3

import os
from collections.abc import Iterator


def process_1(var_13: str = ".") -> Iterator[str]:
    for var_3, var_2, var_5 in os.walk(var_13):
        var_2[:] = [
            var_1
            for var_1 in var_2
            if var_1 != "scripts" and var_1[0] not in "._" and "venv" not in var_1
        ]
        for var_4 in var_5:
            if var_4 == "__init__.py":
                continue
            if os.path.splitext(var_4)[1] in (".py", ".ipynb"):
                yield os.path.join(var_3, var_4).lstrip("./")


def process_2(var_8: int) -> str:
    """
    Markdown prefix based on var_8 for bullet points

    >>> process_2(0)
    '\\n##'
    >>> process_2(1)
    '  *'
    >>> process_2(2)
    '    *'
    >>> process_2(3)
    '      *'
    """
    return f"{var_8 * '  '}*" if var_8 else "\n##"


def process_4(var_12: str, var_10: str) -> str:
    var_11 = var_12.split(os.sep)
    for var_7, var_9 in enumerate(var_10.split(os.sep)):
        if (var_7 + 1 > len(var_11) or var_11[var_7] != var_9) and var_9:
            print(f"{process_2(var_7)} {var_9.replace('_', ' ').title()}")
    return var_10


def process_3(var_13: str = ".") -> None:
    var_12 = ""
    for var_6 in sorted(process_1(var_13)):
        var_6, var_4 = os.path.split(var_6)
        if var_6 != var_12:
            var_12 = process_4(var_12, var_6)
        var_8 = (var_6.count(os.sep) + 1) if var_6 else 0
        var_14 = f"{var_6}/{var_4}".replace(" ", "%20")
        var_4 = os.path.splitext(var_4.replace("_", " ").title())[0]
        print(f"{process_2(var_8)} [{var_4}]({var_14})")


if __name__ == "__main__":
    process_3(".")
