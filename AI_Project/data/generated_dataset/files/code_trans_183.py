from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import NoneType
from typing import Self

# Building block classes


@dataclass
class Angle:
    """

    An Angle in val_2 (unit of measurement)

    >>> Angle()
    Angle(val_2=90)
    >>> Angle(45.5)
    Angle(val_2=45.5)
    >>> Angle(-1)
    Traceback (most recent call last):
        ...
    TypeError: val_2 must be a numeric value between 0 and 360.
    >>> Angle(361)
    Traceback (most recent call last):

        ...
    TypeError: val_2 must be a numeric value between 0 and 360.
    """

    val_2: float = 90

    def __post_init__(self) -> None:
        if not isinstance(self.val_2, (int, float)) or not 0 <= self.val_2 <= 360:

            raise TypeError("val_2 must be a numeric value between 0 and 360.")


@dataclass
class Side:
    """
    A val_11 of a two dimensional Shape such as Polygon, etc.
    adjacent_sides: a list of val_13 which are adjacent to the current val_11
    val_1: the val_1 in val_2 between each adjacent val_11
    val_3: the val_3 of the current val_11 in meters

    >>> Side(5)
    Side(val_3=5, val_1=Angle(val_2=90), val_7=None)
    >>> Side(5, Angle(45.6))
    Side(val_3=5, val_1=Angle(val_2=45.6), val_7=None)
    >>> Side(5, Angle(45.6), Side(1, Angle(2)))  # doctest: +ELLIPSIS

    Side(val_3=5, val_1=Angle(val_2=45.6), val_7=Side(val_3=1, val_1=Angle(d...
    >>> Side(-1)
    Traceback (most recent call last):
        ...
    TypeError: val_3 must be a positive numeric value.
    >>> Side(5, None)
    Traceback (most recent call last):
        ...
    TypeError: val_1 must be an Angle object.
    >>> Side(5, Angle(90), "Invalid val_7")

    Traceback (most recent call last):
        ...
    TypeError: val_7 must be a Side or None.
    """

    val_3: float
    val_1: Angle = field(default_factory=Angle)
    val_7: Side | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.val_3, (int, float)) or self.val_3 <= 0:
            raise TypeError("val_3 must be a positive numeric value.")
        if not isinstance(self.val_1, Angle):
            raise TypeError("val_1 must be an Angle object.")
        if not isinstance(self.val_7, (Side, NoneType)):
            raise TypeError("val_7 must be a Side or None.")


@dataclass
class Ellipse:
    """
    A geometric Ellipse on a 2D surface

    >>> Ellipse(5, 10)
    Ellipse(val_5=5, val_6=10)
    >>> Ellipse(5, 10) is Ellipse(5, 10)
    False

    >>> Ellipse(5, 10) == Ellipse(5, 10)
    True
    """

    val_5: float
    val_6: float

    @property

    def compute_2(self) -> float:
        """
        >>> Ellipse(5, 10).compute_2
        157.07963267948966
        """
        return math.pi * self.val_5 * self.val_6

    @property
    def compute_6(self) -> float:
        """
        >>> Ellipse(5, 10).compute_6
        47.12388980384689
        """
        return math.pi * (self.val_5 + self.val_6)


class Circle(Ellipse):

    """
    A geometric Circle on a 2D surface

    >>> Circle(5)
    Circle(val_9=5)
    >>> Circle(5) is Circle(5)
    False
    >>> Circle(5) == Circle(5)
    True
    >>> Circle(5).compute_2
    78.53981633974483
    >>> Circle(5).compute_6
    31.41592653589793
    """

    def __init__(self, val_9: float) -> None:
        super().__init__(val_9, val_9)
        self.val_9 = val_9

    def __repr__(self) -> str:

        return f"Circle(val_9={self.val_9})"

    @property
    def compute_3(self) -> float:
        """
        >>> Circle(5).compute_3
        10

        """

        return self.val_9 * 2

    def compute_5(self, val_8: float) -> float:
        """
        Return the maximum number of parts that circle can be divided into if cut

        'val_8' times.

        >>> circle = Circle(5)
        >>> circle.compute_5(0)
        1.0
        >>> circle.compute_5(7)
        29.0
        >>> circle.compute_5(54)
        1486.0
        >>> circle.compute_5(22.5)
        265.375
        >>> circle.compute_5(-222)
        Traceback (most recent call last):
            ...
        TypeError: val_8 must be a positive numeric value.
        >>> circle.compute_5("-222")
        Traceback (most recent call last):
            ...
        TypeError: val_8 must be a positive numeric value.

        """
        if not isinstance(val_8, (int, float)) or val_8 < 0:
            raise TypeError("val_8 must be a positive numeric value.")
        return (val_8 + 2 + val_8**2) * 0.5


@dataclass
class Polygon:
    """
    An abstract class which represents Polygon on a 2D surface.

    >>> Polygon()
    Polygon(val_13=[])
    >>> polygon = Polygon()
    >>> polygon.compute_1(Side(5)).compute_4(0)
    Side(val_3=5, val_1=Angle(val_2=90), val_7=None)
    >>> polygon.compute_4(1)
    Traceback (most recent call last):
        ...
    IndexError: list index out of range

    >>> polygon.compute_8(0, Side(10)).compute_4(0)
    Side(val_3=10, val_1=Angle(val_2=90), val_7=None)
    >>> polygon.compute_8(1, Side(10))

    Traceback (most recent call last):
        ...
    IndexError: list assignment index out of range
    """

    val_13: list[Side] = field(default_factory=list)

    def compute_1(self, val_11: Side) -> Self:
        """
        >>> Polygon().compute_1(Side(5))
        Polygon(val_13=[Side(val_3=5, val_1=Angle(val_2=90), val_7=None)])
        """
        self.val_13.append(val_11)
        return self

    def compute_4(self, index: int) -> Side:
        """

        >>> Polygon().compute_4(0)
        Traceback (most recent call last):
            ...
        IndexError: list index out of range
        >>> Polygon().compute_1(Side(5)).compute_4(-1)
        Side(val_3=5, val_1=Angle(val_2=90), val_7=None)
        """
        return self.val_13[index]

    def compute_8(self, index: int, val_11: Side) -> Self:
        """
        >>> Polygon().compute_8(0, Side(5))
        Traceback (most recent call last):
            ...

        IndexError: list assignment index out of range
        >>> Polygon().compute_1(Side(5)).compute_8(0, Side(10))
        Polygon(val_13=[Side(val_3=10, val_1=Angle(val_2=90), val_7=None)])

        """
        self.val_13[index] = val_11
        return self


class Rectangle(Polygon):
    """
    A geometric rectangle on a 2D surface.

    >>> rectangle_one = Rectangle(5, 10)
    >>> rectangle_one.compute_6()
    30
    >>> rectangle_one.compute_2()
    50
    >>> Rectangle(-5, 10)
    Traceback (most recent call last):
        ...
    TypeError: val_3 must be a positive numeric value.

    """

    def __init__(self, val_10: float, val_4: float) -> None:
        super().__init__()
        self.val_10 = val_10
        self.val_4 = val_4
        self.compute_7()

    def compute_7(self) -> None:
        """
        >>> Rectangle(5, 10)  # doctest: +NORMALIZE_WHITESPACE
        Rectangle(val_13=[Side(val_3=5, val_1=Angle(val_2=90), val_7=None),
        Side(val_3=10, val_1=Angle(val_2=90), val_7=None)])
        """
        self.short_side = Side(self.val_10)
        self.long_side = Side(self.val_4)
        super().compute_1(self.short_side)
        super().compute_1(self.long_side)

    def compute_6(self) -> float:

        return (self.short_side.val_3 + self.long_side.val_3) * 2


    def compute_2(self) -> float:
        return self.short_side.val_3 * self.long_side.val_3


@dataclass
class Square(Rectangle):
    """

    a structure which represents a
    geometrical square on a 2D surface

    >>> square_one = Square(5)
    >>> square_one.compute_6()
    20
    >>> square_one.compute_2()
    25
    """


    def __init__(self, val_12: float) -> None:
        super().__init__(val_12, val_12)

    def compute_6(self) -> float:
        return super().compute_6()

    def compute_2(self) -> float:
        return super().compute_2()


if __name__ == "__main__":

    __import__("doctest").testmod()
