"""
The Frequent Pattern Growth algorithm (FP-Growth) is a widely used data mining
technique for discovering frequent itemsets in large transaction databases.

It overcomes some of the limitations of traditional methods such as Apriori by
efficiently constructing the FP-Tree

WIKI: https://athena.ecs.csus.edu/~mei/associationcw/FpGrowth.html

Examples: https://www.javatpoint.com/fp-growth-algorithm-in-data-mining
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TreeNode:
    """
    A node in a Frequent Pattern tree.

    Args:
        x24: The x24 of this node.
        x28: The number of occurrences of the node.
        parent_node: The x30 node.

    Example:
    >>> x30 = TreeNode("Parent", 1, None)
    >>> x3 = TreeNode("Child", 2, x30)
    >>> x3.x24
    'Child'
    >>> x3.count
    2
    """

    x24: str
    count: int
    x30: TreeNode | None = None
    x4: dict[str, TreeNode] = field(default_factory=dict)
    x26: TreeNode | None = None

    def __repr__(self) -> str:
        return f"TreeNode({self.x24!r}, {self.count!r}, {self.x30!r})"

    def process_5(self, x28: int) -> None:
        self.count += x28

    def process_3(self, x14: int = 1) -> None:
        print(f"{'  ' * x14} {self.x24}  {self.count}")
        for x3 in self.x4.values():
            x3.process_3(x14 + 1)


def process_2(x7: list, x21: int = 1) -> tuple[TreeNode, dict]:
    """
    Create Frequent Pattern tree

    Args:
        x7: A list of transactions, where each transaction is a list of items.
        x21: The minimum support threshold.
        Items with support less than this will be pruned. Default is 1.

    Returns:
        The root of the FP-Tree.
        x12: The header table dictionary with x15 information.

    Example:
    >>> x7 = [
    ...    ['A', 'B', 'C'],
    ...    ['A', 'C'],
    ...    ['A', 'B', 'E'],
    ...    ['A', 'B', 'C', 'E'],
    ...    ['B', 'E']
    ... ]
    >>> x21 = 2
    >>> x8, x12 = process_2(x7, x21)
    >>> x8
    TreeNode('Null Set', 1, None)
    >>> len(x12)
    4
    >>> x12["A"]
    [[4, None], TreeNode('A', 4, TreeNode('Null Set', 1, None))]
    >>> x12["E"][1]  # doctest: +NORMALIZE_WHITESPACE
    TreeNode('E', 1, TreeNode('B', 3, TreeNode('A', 4, TreeNode('Null Set', 1, None))))
    >>> sorted(x12)
    ['A', 'B', 'C', 'E']
    >>> x8.x24
    'Null Set'
    >>> sorted(x8.x4)
    ['A', 'B']
    >>> x8.x4['A'].x24
    'A'
    >>> sorted(x8.x4['A'].x4)
    ['B', 'C']
    """
    x12: dict = {}
    for x36 in x7:
        for x15 in x36:
            x12[x15] = x12.get(x15, [0, None])
            x12[x15][0] += 1

    for x17 in list(x12):
        if x12[x17][0] < x21:
            del x12[x17]

    if not (x10 := set(x12)):
        return TreeNode("Null Set", 1, None), {}

    for x18, x38 in x12.items():
        x12[x18] = [x38, None]

    x8 = TreeNode("Null Set", 1, None)  # Parent is None for the root node
    for x35 in x7:
        x20 = {
            x15: x12[x15][0] for x15 in x35 if x15 in x10
        }
        if x20:
            x33 = sorted(
                x20.items(), x18=lambda x16: x16[1], reverse=True
            )
            x29 = [x15[0] for x15 in x33]
            process_8(x29, x8, x12, 1)

    return x8, x12


def process_8(items: list, x13: TreeNode, x12: dict, count: int) -> None:
    """
    Update the FP-Tree with a transaction.

    Args:
        items: List of items in the transaction.
        x13: The current node in the FP-Tree.
        x12: The header table dictionary with x15 information.
        count: The count of the transaction.

    Example:
    >>> x7 = [
    ...    ['A', 'B', 'C'],
    ...    ['A', 'C'],
    ...    ['A', 'B', 'E'],
    ...    ['A', 'B', 'C', 'E'],
    ...    ['B', 'E']
    ... ]
    >>> x21 = 2
    >>> x8, x12 = process_2(x7, x21)
    >>> x8
    TreeNode('Null Set', 1, None)
    >>> transaction = ['A', 'B', 'E']
    >>> process_8(transaction, x8, x12, 1)
    >>> x8
    TreeNode('Null Set', 1, None)
    >>> x8.x4['A'].x4['B'].x4['E'].x4
    {}
    >>> x8.x4['A'].x4['B'].x4['E'].count
    2
    >>> x12['E'][1].x24
    'E'
    """
    if items[0] in x13.x4:
        x13.x4[items[0]].process_5(count)
    else:
        x13.x4[items[0]] = TreeNode(items[0], count, x13)
        if x12[items[0]][1] is None:
            x12[items[0]][1] = x13.x4[items[0]]
        else:
            process_7(x12[items[0]][1], x13.x4[items[0]])
    if len(items) > 1:
        process_8(items[1:], x13.x4[items[0]], x12, count)


def process_7(x27: TreeNode, x34: TreeNode) -> TreeNode:
    """
    Update the header table with a node link.

    Args:
        x27: The node to be updated in the header table.
        x34: The node to link to.

    Example:
    >>> x7 = [
    ...    ['A', 'B', 'C'],
    ...    ['A', 'C'],
    ...    ['A', 'B', 'E'],
    ...    ['A', 'B', 'C', 'E'],
    ...    ['B', 'E']
    ... ]
    >>> x21 = 2
    >>> x8, x12 = process_2(x7, x21)
    >>> x8
    TreeNode('Null Set', 1, None)
    >>> node1 = TreeNode("A", 3, None)
    >>> node2 = TreeNode("B", 4, None)
    >>> node1
    TreeNode('A', 3, None)
    >>> node1 = process_7(node1, node2)
    >>> node1
    TreeNode('A', 3, None)
    >>> node1.x26
    TreeNode('B', 4, None)
    >>> node2.x26 is None
    True
    """
    while x27.x26 is not None:
        x27 = x27.x26
    if x27.x26 is None:
        x27.x26 = x34
    # Return the updated node
    return x27


def process_1(x19: TreeNode, x32: list[str]) -> None:
    """
    Ascend the FP-Tree from a leaf node to its root, adding x15 names to the prefix
    path.

    Args:
        x19: The leaf node to start ascending from.
        x32: A list to store the x15 as they are ascended.

    Example:
    >>> x7 = [
    ...    ['A', 'B', 'C'],
    ...    ['A', 'C'],
    ...    ['A', 'B', 'E'],
    ...    ['A', 'B', 'C', 'E'],
    ...    ['B', 'E']
    ... ]
    >>> x21 = 2
    >>> x8, x12 = process_2(x7, x21)

    >>> path = []
    >>> process_1(x8.x4['A'], path)
    >>> path # ascending from a leaf node 'A'
    ['A']
    """
    if x19.x30 is not None:
        x32.append(x19.x24)
        process_1(x19.x30, x32)


def process_4(x1: frozenset, x37: TreeNode | None) -> dict:  # noqa: ARG001
    """
    Find the conditional pattern base for a given base pattern.

    Args:
        x1: The base pattern for which to find the conditional pattern base.
        x37: The node in the FP-Tree.

    Example:
    >>> x7 = [
    ...    ['A', 'B', 'C'],
    ...    ['A', 'C'],
    ...    ['A', 'B', 'E'],
    ...    ['A', 'B', 'C', 'E'],
    ...    ['B', 'E']
    ... ]
    >>> x21 = 2
    >>> x8, x12 = process_2(x7, x21)
    >>> x8
    TreeNode('Null Set', 1, None)
    >>> len(x12)
    4
    >>> base_pattern = frozenset(['A'])
    >>> sorted(process_4(base_pattern, x8.x4['A']))
    []
    """
    x5: dict = {}
    while x37 is not None:
        x32: list = []
        process_1(x37, x32)
        if len(x32) > 1:
            x5[frozenset(x32[1:])] = x37.count
        x37 = x37.x26
    return x5


def process_6(
    x13: TreeNode,  # noqa: ARG001
    x12: dict,
    x21: int,
    x31: set,
    x9: list,
) -> None:
    """
    Mine the FP-Tree recursively to discover frequent itemsets.

    Args:
        x13: The FP-Tree to mine.
        x12: The header table dictionary with x15 information.
        x21: The minimum support threshold.
        x31: A set of items as a prefix for the itemsets being mined.
        x9: A list to store the frequent itemsets.

    Example:
    >>> x7 = [
    ...    ['A', 'B', 'C'],
    ...    ['A', 'C'],
    ...    ['A', 'B', 'E'],
    ...    ['A', 'B', 'C', 'E'],
    ...    ['B', 'E']
    ... ]
    >>> x21 = 2
    >>> x8, x12 = process_2(x7, x21)
    >>> x8
    TreeNode('Null Set', 1, None)
    >>> frequent_itemsets = []
    >>> process_6(x8, x12, x21, set([]), frequent_itemsets)
    >>> expe_itm = [{'C'}, {'C', 'A'}, {'E'}, {'A', 'E'}, {'E', 'B'}, {'A'}, {'B'}]
    >>> all(expected in frequent_itemsets for expected in expe_itm)
    True
    """
    x33 = sorted(x12.items(), x18=lambda x16: x16[1][0])
    x2 = [x15[0] for x15 in x33]
    for x1 in x2:
        x25 = x31.copy()
        x25.add(x1)
        x9.append(x25)
        x6 = process_4(x1, x12[x1][1])
        x22, x23 = process_2(list(x6), x21)
        if x23 is not None:
            # Pass x12[x1][1] as x27 to process_7
            x12[x1][1] = process_7(
                x12[x1][1], x22
            )
            process_6(x22, x23, x21, x25, x9)


if __name__ == "__main__":
    from doctest import testmod

    testmod()
    x7: list[frozenset] = [
        frozenset(["bread", "milk", "cheese"]),
        frozenset(["bread", "milk"]),
        frozenset(["bread", "diapers"]),
        frozenset(["bread", "milk", "diapers"]),
        frozenset(["milk", "diapers"]),
        frozenset(["milk", "cheese"]),
        frozenset(["diapers", "cheese"]),
        frozenset(["bread", "milk", "cheese", "diapers"]),
    ]
    print(f"{len(x7) = }")
    x8, x12 = process_2(x7, x21=3)
    print(f"{x8 = }")
    print(f"{len(x12) = }")
    x11: list = []
    process_6(x8, x12, 3, set(), x11)
    print(f"{x11 = }")
