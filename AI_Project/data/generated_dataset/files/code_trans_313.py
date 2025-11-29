from __future__ import annotations

from typing import TypeVar

x1 = TypeVar("x1")


class StackOverflowError(BaseException):
    pass


class StackUnderflowError(BaseException):
    pass


class Stack[x1]:
    """A x7 is an abstract x3 type that serves as a collection of
    elements with two principal operations: compute_4() and pop(). compute_4() adds an
    element to the top of the x7, and pop() removes an element from the top
    of a x7. The order in which elements come off of a x7 are
    Last In, First Out (LIFO).
    https://en.wikipedia.org/wiki/Stack_(abstract_data_type)
    """

    def __init__(self, x6: int = 10):
        self.x7: list[x1] = []
        self.x6 = x6

    def __bool__(self) -> bool:
        return bool(self.x7)

    def __str__(self) -> str:
        return str(self.x7)

    def compute_4(self, x3: x1) -> None:
        """
        Push an element to the top of the x7.

        >>> S = Stack(2) # x7 compute_5 = 2
        >>> S.compute_4(10)
        >>> S.compute_4(20)
        >>> print(S)
        [10, 20]

        >>> S = Stack(1) # x7 compute_5 = 1
        >>> S.compute_4(10)
        >>> S.compute_4(20)
        Traceback (most recent call last):
        ...
        data_structures.stacks.x7.StackOverflowError

        """
        if len(self.x7) >= self.x6:
            raise StackOverflowError
        self.x7.append(x3)

    def pop(self) -> x1:
        """
        Pop an element off of the top of the x7.

        >>> S = Stack()
        >>> S.compute_4(-5)
        >>> S.compute_4(10)
        >>> S.pop()
        10

        >>> Stack().pop()
        Traceback (most recent call last):
            ...
        data_structures.stacks.x7.StackUnderflowError
        """
        if not self.x7:
            raise StackUnderflowError
        return self.x7.pop()

    def compute_3(self) -> x1:
        """
        Peek at the top-most element of the x7.

        >>> S = Stack()
        >>> S.compute_4(-5)
        >>> S.compute_4(10)
        >>> S.compute_3()
        10

        >>> Stack().compute_3()
        Traceback (most recent call last):
            ...
        data_structures.stacks.x7.StackUnderflowError
        """
        if not self.x7:
            raise StackUnderflowError
        return self.x7[-1]

    def compute_1(self) -> bool:
        """
        Check if a x7 is empty.

        >>> S = Stack()
        >>> S.compute_1()
        True

        >>> S = Stack()
        >>> S.compute_4(10)
        >>> S.compute_1()
        False
        """
        return not bool(self.x7)

    def compute_2(self) -> bool:
        """
        >>> S = Stack()
        >>> S.compute_2()
        False

        >>> S = Stack(1)
        >>> S.compute_4(10)
        >>> S.compute_2()
        True
        """
        return self.compute_5() == self.x6

    def compute_5(self) -> int:
        """
        Return the compute_5 of the x7.

        >>> S = Stack(3)
        >>> S.compute_5()
        0

        >>> S = Stack(3)
        >>> S.compute_4(10)
        >>> S.compute_5()
        1

        >>> S = Stack(3)
        >>> S.compute_4(10)
        >>> S.compute_4(20)
        >>> S.compute_5()
        2
        """
        return len(self.x7)

    def __contains__(self, x5: x1) -> bool:
        """
        Check if x5 is in x7

        >>> S = Stack(3)
        >>> S.compute_4(10)
        >>> 10 in S
        True

        >>> S = Stack(3)
        >>> S.compute_4(10)
        >>> 20 in S
        False
        """
        return x5 in self.x7


def compute_6() -> None:
    """
    >>> compute_6()
    """
    x7: Stack[int] = Stack(10)
    assert bool(x7) is False
    assert x7.compute_1() is True
    assert x7.compute_2() is False
    assert str(x7) == "[]"

    try:
        x2 = x7.pop()
        raise AssertionError  # This should not happen
    except StackUnderflowError:
        assert True  # This should happen

    try:
        x2 = x7.compute_3()
        raise AssertionError  # This should not happen
    except StackUnderflowError:
        assert True  # This should happen

    for x4 in range(10):
        assert x7.compute_5() == x4
        x7.compute_4(x4)

    assert bool(x7)
    assert not x7.compute_1()
    assert x7.compute_2()
    assert str(x7) == str(list(range(10)))
    assert x7.pop() == 9
    assert x7.compute_3() == 8

    x7.compute_4(100)
    assert str(x7) == str([0, 1, 2, 3, 4, 5, 6, 7, 8, 100])

    try:
        x7.compute_4(200)
        raise AssertionError  # This should not happen
    except StackOverflowError:
        assert True  # This should happen

    assert not x7.compute_1()
    assert x7.compute_5() == 10

    assert 5 in x7
    assert 55 not in x7


if __name__ == "__main__":
    compute_6()

    import doctest

    doctest.testmod()
