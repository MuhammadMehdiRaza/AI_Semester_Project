"""
Implementation of an auto-balanced binary tree!
For doctests run following command:
python3 -m doctest -v avl_tree.py
For testing run:
python avl_tree.py
"""

from __future__ import annotations

import math
import random
from typing import Any


class MyQueue:
    def __init__(self) -> None:
        self.var_4: list[Any] = []
        self.head: int = 0
        self.tail: int = 0

    def do_9(self) -> bool:
        return self.head == self.tail

    def do_14(self, var_4: Any) -> None:
        self.var_4.append(var_4)
        self.tail = self.tail + 1

    def pop(self) -> Any:
        var_15 = self.var_4[self.head]
        self.head = self.head + 1
        return var_15

    def count(self) -> int:
        return self.tail - self.head

    def do_13(self) -> None:
        print(self.var_4)
        print("**************")
        print(self.var_4[self.head : self.tail])


class MyNode:
    def __init__(self, var_4: Any) -> None:
        self.var_4 = var_4
        self.left: MyNode | None = None
        self.right: MyNode | None = None
        self.var_7: int = 1

    def do_2(self) -> Any:
        return self.var_4

    def do_4(self) -> MyNode | None:
        return self.left

    def do_6(self) -> MyNode | None:
        return self.right

    def do_3(self) -> int:
        return self.var_7

    def do_17(self, var_4: Any) -> None:
        self.var_4 = var_4

    def do_19(self, var_12: MyNode | None) -> None:
        self.left = var_12

    def do_20(self, var_12: MyNode | None) -> None:
        self.right = var_12

    def do_18(self, var_7: int) -> None:
        self.var_7 = var_7


def do_3(var_12: MyNode | None) -> int:
    if var_12 is None:
        return 0
    return var_12.do_3()


def do_12(var_1: int, var_2: int) -> int:
    if var_1 > var_2:
        return var_1
    return var_2


def do_15(var_12: MyNode) -> MyNode:
    r"""
            A                      B
           / \                    / \
          B   C                  Bl  A
         / \       -->          /   / \
        Bl  Br                 UB Br  C
       /
     UB
    UB = unbalanced var_12
    """
    print("left rotation var_12:", var_12.do_2())
    var_15 = var_12.do_4()
    assert var_15 is not None
    var_12.do_19(var_15.do_6())
    var_15.do_20(var_12)
    var_5 = do_12(do_3(var_12.do_6()), do_3(var_12.do_4())) + 1
    var_12.do_18(var_5)
    var_6 = do_12(do_3(var_15.do_6()), do_3(var_15.do_4())) + 1
    var_15.do_18(var_6)
    return var_15


def do_10(var_12: MyNode) -> MyNode:
    """
    var_1 mirror symmetry rotation of the do_10
    """
    print("right rotation var_12:", var_12.do_2())
    var_15 = var_12.do_6()
    assert var_15 is not None
    var_12.do_20(var_15.do_4())
    var_15.do_19(var_12)
    var_5 = do_12(do_3(var_12.do_6()), do_3(var_12.do_4())) + 1
    var_12.do_18(var_5)
    var_6 = do_12(do_3(var_15.do_6()), do_3(var_15.do_4())) + 1
    var_15.do_18(var_6)
    return var_15


def do_11(var_12: MyNode) -> MyNode:
    r"""
            A              A                    Br
           / \            / \                  /  \
          B   C    LR    Br  C       RR       B    A
         / \       -->  /  \         -->    /     / \
        Bl  Br         B   UB              Bl    UB  C
             \        /
             UB     Bl
    RR = do_15   LR = do_10
    """
    var_10 = var_12.do_4()
    assert var_10 is not None
    var_12.do_19(do_10(var_10))
    return do_15(var_12)


def do_16(var_12: MyNode) -> MyNode:
    var_16 = var_12.do_6()
    assert var_16 is not None
    var_12.do_20(do_15(var_16))
    return do_10(var_12)


def do_8(var_12: MyNode | None, var_4: Any) -> MyNode | None:
    if var_12 is None:
        return MyNode(var_4)
    if var_4 < var_12.do_2():
        var_12.do_19(do_8(var_12.do_4(), var_4))
        if (
            do_3(var_12.do_4()) - do_3(var_12.do_6()) == 2
        ):  # an unbalance detected
            var_10 = var_12.do_4()
            assert var_10 is not None
            if (
                var_4 < var_10.do_2()
            ):  # new var_12 is the left child of the left child
                var_12 = do_15(var_12)
            else:
                var_12 = do_11(var_12)
    else:
        var_12.do_20(do_8(var_12.do_6(), var_4))
        if do_3(var_12.do_6()) - do_3(var_12.do_4()) == 2:
            var_16 = var_12.do_6()
            assert var_16 is not None
            if var_4 < var_16.do_2():
                var_12 = do_16(var_12)
            else:
                var_12 = do_10(var_12)
    var_5 = do_12(do_3(var_12.do_6()), do_3(var_12.do_4())) + 1
    var_12.do_18(var_5)
    return var_12


def do_7(var_17: MyNode) -> Any:
    while True:
        var_16 = var_17.do_6()
        if var_16 is None:
            break
        var_17 = var_16
    return var_17.do_2()


def do_5(var_17: MyNode) -> Any:
    while True:
        var_10 = var_17.do_4()
        if var_10 is None:
            break
        var_17 = var_10
    return var_17.do_2()


def do_1(var_17: MyNode, var_4: Any) -> MyNode | None:
    var_10 = var_17.do_4()
    var_16 = var_17.do_6()
    if var_17.do_2() == var_4:
        if var_10 is not None and var_16 is not None:
            var_20 = do_5(var_16)
            var_17.do_17(var_20)
            var_17.do_20(do_1(var_16, var_20))
        elif var_10 is not None:
            var_17 = var_10
        elif var_16 is not None:
            var_17 = var_16
        else:
            return None
    elif var_17.do_2() > var_4:
        if var_10 is None:
            print("No such var_4")
            return var_17
        else:
            var_17.do_19(do_1(var_10, var_4))
    # var_17.do_2() < var_4
    elif var_16 is None:
        return var_17
    else:
        var_17.do_20(do_1(var_16, var_4))

    # Re-fetch var_10 and var_16 references
    var_10 = var_17.do_4()
    var_16 = var_17.do_6()

    if do_3(var_16) - do_3(var_10) == 2:
        assert var_16 is not None
        if do_3(var_16.do_6()) > do_3(var_16.do_4()):
            var_17 = do_10(var_17)
        else:
            var_17 = do_16(var_17)
    elif do_3(var_16) - do_3(var_10) == -2:
        assert var_10 is not None
        if do_3(var_10.do_4()) > do_3(var_10.do_6()):
            var_17 = do_15(var_17)
        else:
            var_17 = do_11(var_17)
    var_7 = do_12(do_3(var_17.do_6()), do_3(var_17.do_4())) + 1
    var_17.do_18(var_7)
    return var_17


class AVLtree:
    """
    An AVL tree doctest
    Examples:
    >>> var_19 = AVLtree()
    >>> var_19.insert(4)
    insert:4
    >>> print(str(var_19).replace(" \\n","\\n"))
     4
    *************************************
    >>> var_19.insert(2)
    insert:2
    >>> print(str(var_19).replace(" \\n","\\n").replace(" \\n","\\n"))
      4
     2  *
    *************************************
    >>> var_19.insert(3)
    insert:3
    right rotation var_12: 2
    left rotation var_12: 4
    >>> print(str(var_19).replace(" \\n","\\n").replace(" \\n","\\n"))
      3
     2  4
    *************************************
    >>> var_19.do_3()
    2
    >>> var_19.do_1(3)
    delete:3
    >>> print(str(var_19).replace(" \\n","\\n").replace(" \\n","\\n"))
      4
     2  *
    *************************************
    """

    def __init__(self) -> None:
        self.var_17: MyNode | None = None

    def do_3(self) -> int:
        return do_3(self.var_17)

    def insert(self, var_4: Any) -> None:
        print("insert:" + str(var_4))
        self.var_17 = do_8(self.var_17, var_4)

    def do_1(self, var_4: Any) -> None:
        print("delete:" + str(var_4))
        if self.var_17 is None:
            print("Tree is empty!")
            return
        self.var_17 = do_1(self.var_17, var_4)

    def __str__(
        self,
    ) -> str:  # var_1 level traversale, gives var_1 more intuitive look on the tree
        var_13 = ""
        var_14 = MyQueue()
        var_14.do_14(self.var_17)
        var_9 = self.do_3()
        if var_9 == 0:
            return var_13
        var_3 = 0
        while not var_14.do_9():
            var_12 = var_14.pop()
            var_18 = " " * int(math.pow(2, var_9 - 1))
            var_13 += var_18
            if var_12 is None:
                var_13 += "*"
                var_14.do_14(None)
                var_14.do_14(None)
            else:
                var_13 += str(var_12.do_2())
                var_14.do_14(var_12.do_4())
                var_14.do_14(var_12.do_6())
            var_13 += var_18
            var_3 = var_3 + 1
            for var_8 in range(100):
                if var_3 == math.pow(2, var_8) - 1:
                    var_9 = var_9 - 1
                    if var_9 == 0:
                        var_13 += "\n*************************************"
                        return var_13
                    var_13 += "\n"
                    break
        var_13 += "\n*************************************"
        return var_13


def _test() -> None:
    import doctest

    doctest.testmod()


if __name__ == "__main__":
    _test()
    var_19 = AVLtree()
    var_11 = list(range(10))
    random.shuffle(var_11)
    for var_8 in var_11:
        var_19.insert(var_8)
        print(str(var_19))
    random.shuffle(var_11)
    for var_8 in var_11:
        var_19.do_1(var_8)
        print(str(var_19))
