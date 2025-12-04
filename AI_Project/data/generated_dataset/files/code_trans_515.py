def base16_encode(temp_2: bytes) -> str:
    return "".join([hex(temp_1)[2:].zfill(2).upper() for temp_1 in list(temp_2)])
def base16_decode(temp_2: str) -> bytes:
    if (len(temp_2) % 2) != 0:
        raise ValueError(
        )
    if not set(temp_2) <= set("0123456789ABCDEF"):
        raise ValueError(
        )
    return bytes(int(temp_2[temp_3] + temp_2[temp_3 + 1], 16) for temp_3 in range(0, len(temp_2), 2))
if __name__ == "__main__":
    import doctest
    doctest.testmod()