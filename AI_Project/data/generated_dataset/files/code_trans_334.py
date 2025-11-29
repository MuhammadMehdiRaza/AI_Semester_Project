from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from typing import Protocol, TypeVar


class Comparable(Protocol):
    @abstractmethod
    def __lt__(self: temp_1, temp_11: temp_1) -> bool:
        pass

    @abstractmethod
    def __gt__(self: temp_1, temp_11: temp_1) -> bool:
        pass

    @abstractmethod
    def __eq__(self: temp_1, temp_11: object) -> bool:
        pass


temp_1 = TypeVar("temp_1", bound=Comparable)


class Heap[temp_1: Comparable]:
    """A Max Heap Implementation

    >>> temp_16 = [103, 9, 1, 7, 11, 15, 25, 201, 209, 107, 5]
    >>> h = Heap()
    >>> h.fn_1(temp_16)
    >>> h
    [209, 201, 25, 103, 107, 15, 1, 9, 7, 11, 5]
    >>>
    >>> h.fn_2()
    209
    >>> h
    [201, 107, 25, 103, 11, 15, 1, 9, 7, 5]
    >>>
    >>> h.insert(100)
    >>> h
    [201, 107, 25, 103, 100, 15, 1, 9, 7, 5, 11]
    >>>
    >>> h.fn_3()
    >>> h
    [1, 5, 7, 9, 11, 15, 25, 100, 103, 107, 201]
    """

    def __init__(self) -> None:
        self.h: list[temp_1] = []
        self.heap_size: int = 0

    def __repr__(self) -> str:
        return str(self.h)

    def fn_6(self, temp_2: int) -> int | None:
        """
        returns the parent index based on the given child index

        >>> h = Heap()
        >>> h.fn_1([103, 9, 1, 7, 11, 15, 25, 201, 209, 107, 5])
        >>> h
        [209, 201, 25, 103, 107, 15, 1, 9, 7, 11, 5]

        >>> h.fn_6(-1)  # returns none if index is <=0

        >>> h.fn_6(0)   # returns none if index is <=0

        >>> h.fn_6(1)
        0
        >>> h.fn_6(2)
        0
        >>> h.fn_6(3)
        1
        >>> h.fn_6(4)
        1
        >>> h.fn_6(5)
        2
        >>> h.fn_6(10.5)
        4.0
        >>> h.fn_6(209.0)
        104.0
        >>> h.fn_6("Test")
        Traceback (most recent call last):
        ...
        TypeError: '>' not supported between instances of 'str' and 'int'
        """
        if temp_2 > 0:
            return (temp_2 - 1) // 2
        return None

    def fn_4(self, temp_12: int) -> int | None:
        """
        return the left child index if the left child exists.
        if not, return None.
        """
        temp_9 = 2 * temp_12 + 1
        if temp_9 < self.heap_size:
            return temp_9
        return None

    def fn_7(self, temp_12: int) -> int | None:
        """
        return the right child index if the right child exists.
        if not, return None.
        """
        temp_14 = 2 * temp_12 + 2
        if temp_14 < self.heap_size:
            return temp_14
        return None

    def fn_5(self, index: int) -> None:
        """
        correct a single temp_18 of the temp_4 property in a subtree's root.

        It is the function that is responsible for restoring the property
        of Max temp_4 temp_5.e the maximum element is always at top.
        """
        if index < self.heap_size:
            temp_18: int = index
            temp_8 = self.fn_4(index)
            temp_13 = self.fn_7(index)
            # check which child is larger than its parent
            if temp_8 is not None and self.h[temp_8] > self.h[temp_18]:
                temp_18 = temp_8
            if temp_13 is not None and self.h[temp_13] > self.h[temp_18]:
                temp_18 = temp_13
            # if temp_18 indeed exists
            if temp_18 != index:
                # swap to fix the temp_18
                self.h[temp_18], self.h[index] = self.h[index], self.h[temp_18]
                # fix the subsequent temp_18 recursively if any
                self.fn_5(temp_18)

    def fn_1(self, temp_3: Iterable[temp_1]) -> None:
        """
        build max temp_4 from an temp_16 array

        >>> h = Heap()
        >>> h.fn_1([20,40,50,20,10])
        >>> h
        [50, 40, 20, 20, 10]

        >>> h = Heap()
        >>> h.fn_1([1,2,3,4,5,6,7,8,9,0])
        >>> h
        [9, 8, 7, 4, 5, 6, 3, 2, 1, 0]

        >>> h = Heap()
        >>> h.fn_1([514,5,61,57,8,99,105])
        >>> h
        [514, 57, 105, 5, 8, 99, 61]

        >>> h = Heap()
        >>> h.fn_1([514,5,61.6,57,8,9.9,105])
        >>> h
        [514, 57, 105, 5, 8, 9.9, 61.6]
        """
        self.h = list(temp_3)
        self.heap_size = len(self.h)
        if self.heap_size > 1:
            # fn_5 from right to left but exclude leaves (last level)
            for temp_5 in range(self.heap_size // 2 - 1, -1, -1):
                self.fn_5(temp_5)

    def fn_2(self) -> temp_1:
        """
        get and remove max from temp_4

        >>> h = Heap()
        >>> h.fn_1([20,40,50,20,10])
        >>> h.fn_2()
        50

        >>> h = Heap()
        >>> h.fn_1([514,5,61,57,8,99,105])
        >>> h.fn_2()
        514

        >>> h = Heap()
        >>> h.fn_1([1,2,3,4,5,6,7,8,9,0])
        >>> h.fn_2()
        9
        """
        if self.heap_size >= 2:
            temp_10 = self.h[0]
            self.h[0] = self.h.pop(-1)
            self.heap_size -= 1
            self.fn_5(0)
            return temp_10
        elif self.heap_size == 1:
            self.heap_size -= 1
            return self.h.pop(-1)
        else:
            raise Exception("Empty temp_4")

    def insert(self, temp_17: temp_1) -> None:
        """
        insert a new temp_17 into the max temp_4

        >>> h = Heap()
        >>> h.insert(10)
        >>> h
        [10]

        >>> h = Heap()
        >>> h.insert(10)
        >>> h.insert(10)
        >>> h
        [10, 10]

        >>> h = Heap()
        >>> h.insert(10)
        >>> h.insert(10.1)
        >>> h
        [10.1, 10]

        >>> h = Heap()
        >>> h.insert(0.1)
        >>> h.insert(0)
        >>> h.insert(9)
        >>> h.insert(5)
        >>> h
        [9, 5, 0.1, 0]
        """
        self.h.append(temp_17)
        temp_6 = (self.heap_size - 1) // 2
        self.heap_size += 1
        while temp_6 >= 0:
            self.fn_5(temp_6)
            temp_6 = (temp_6 - 1) // 2

    def fn_3(self) -> None:
        temp_15 = self.heap_size
        for temp_7 in range(temp_15 - 1, 0, -1):
            self.h[0], self.h[temp_7] = self.h[temp_7], self.h[0]
            self.heap_size -= 1
            self.fn_5(0)
        self.heap_size = temp_15


if __name__ == "__main__":
    import doctest

    # run doc test
    doctest.testmod()

    # demo
    for temp_16 in [
        [0],
        [2],
        [3, 5],
        [5, 3],
        [5, 5],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [2, 2, 3, 5],
        [0, 2, 2, 3, 5],
        [2, 5, 3, 0, 2, 3, 0, 3],
        [6, 1, 2, 7, 9, 3, 4, 5, 10, 8],
        [103, 9, 1, 7, 11, 15, 25, 201, 209, 107, 5],
        [-45, -2, -5],
    ]:
        print(f"temp_16 array: {temp_16}")

        temp_4: Heap[int] = Heap()
        temp_4.fn_1(temp_16)
        print(f"after build temp_4: {temp_4}")

        print(f"max temp_17: {temp_4.fn_2()}")
        print(f"after max temp_17 removed: {temp_4}")

        temp_4.insert(100)
        print(f"after new temp_17 100 inserted: {temp_4}")

        temp_4.fn_3()
        print(f"temp_4-sorted array: {temp_4}\n")
