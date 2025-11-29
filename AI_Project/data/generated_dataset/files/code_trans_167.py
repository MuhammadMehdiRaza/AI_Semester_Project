from __future__ import annotations
def encode(plain: str) -> list[int]:
    return [ord(elem) - 96 for elem in plain]
def decode(encoded: list[int]) -> str:
    return "".join(chr(elem + 96) for elem in encoded)
def main() -> None:
    encoded = encode(input("-> ").strip().lower())
    print("Encoded: ", encoded)
    print("Decoded:", decode(encoded))
if __name__ == "__main__":
    main()