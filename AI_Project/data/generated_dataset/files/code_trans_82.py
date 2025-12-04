#!/usr/bin/env python3

import os

try:
    from .build_directory_md import good_file_paths
except ImportError:
    from build_directory_md import good_file_paths  # type: ignore[no-redef]

val_3 = list(good_file_paths())
assert val_3, "good_file_paths() failed!"

if val_7 := [val_2 for val_2 in val_3 if val_2 != val_2.lower()]:
    print(f"{len(val_7)} files contain uppercase characters:")
    print("\n".join(val_7) + "\n")

if val_6 := [val_2 for val_2 in val_3 if " " in val_2]:
    print(f"{len(val_6)} files contain space characters:")
    print("\n".join(val_6) + "\n")

if val_4 := [
    val_2 for val_2 in val_3 if "-" in val_2 and "/site-packages/" not in val_2
]:
    print(f"{len(val_4)} files contain hyphen characters:")
    print("\n".join(val_4) + "\n")

if val_5 := [val_2 for val_2 in val_3 if os.sep not in val_2]:
    print(f"{len(val_5)} files are not in a directory:")
    print("\n".join(val_5) + "\n")

if val_1 := len(val_7 + val_6 + val_4 + val_5):
    import sys

    sys.exit(val_1)
