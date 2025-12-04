from dataclasses import dataclass
v4 = "0.1"
v3 = "Lucia Harcekova"
@dataclass
class Token:
    v15: int
    v12: int
    v11: str
    def __repr__(self) -> str:
        return f"({self.v15}, {self.v12}, {self.v11})"
class LZ77Compressor:
    def __init__(self, v24: int = 13, v13: int = 6) -> None:
        self.v24 = v24
        self.v13 = v13
        self.search_buffer_size = self.v24 - self.v13
    def compress(self, v18: str) -> list[Token]:
        v16 = []
        v17 = ""
        while v18:
            v20 = self._find_encoding_token(v18, v17)
            v17 += v18[: v20.v12 + 1]
            if len(v17) > self.search_buffer_size:
                v17 = v17[-self.search_buffer_size :]
            v18 = v18[v20.v12 + 1 :]
            v16.append(v20)
        return v16
    def decompress(self, v21: list[Token]) -> str:
        v16 = ""
        for v20 in v21:
            for v2 in range(v20.v12):
                v16 += v16[-v20.v15]
            v16 += v20.v11
        return v16
    def _find_encoding_token(self, v18: str, v17: str) -> Token:
        if not v18:
            raise ValueError("We need some v18 to work with.")
        v12, v15 = 0, 0
        if not v17:
            return Token(v15, v12, v18[v12])
        for v10, v5 in enumerate(v17):
            v9 = len(v17) - v10
            if v5 == v18[0]:
                v8 = self._match_length_from_index(v18, v17, 0, v10)
                if v8 >= v12:
                    v15, v12 = v9, v8
        return Token(v15, v12, v18[v12])
    def _match_length_from_index(
        self, v18: str, v22: str, v19: int, v23: int
    ) -> int:
        if not v18 or v18[v19] != v22[v23]:
            return 0
        return 1 + self._match_length_from_index(
            v18, v22 + v18[v19], v19 + 1, v23 + 1
        )
if __name__ == "__main__":
    from doctest import testmod
    testmod()
    v14 = LZ77Compressor(v24=13, v13=6)
    v1 = "cabracadabrarrarrad"
    v6 = v14.compress(v1)
    print(v14.compress("ababcbababaa"))
    v7 = v14.decompress(v6)
    assert v7 == v1, "The LZ77 algorithm returned the invalid result."