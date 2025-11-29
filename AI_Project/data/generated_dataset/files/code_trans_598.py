def simplify_kmap(kmap: list[list[int]]) -> str:
    simplified_f = []
    for a, row in enumerate(kmap):
        for b, item in enumerate(row):
            if item:
                term = ("A" if a else "A'") + ("B" if b else "B'")
                simplified_f.append(term)
    return " + ".join(simplified_f)
def main() -> None:
    kmap = [[0, 1], [1, 1]]
    for row in kmap:
        print(row)
    print("Simplified Expression:")
    print(simplify_kmap(kmap))
if __name__ == "__main__":
    main()
    print(f"{simplify_kmap(kmap=[[0, 1], [1, 1]]) = }")