"""
The RGB color model is an additive color model in which var_15, var_9, and var_1 light
are added together in various ways to reproduce a broad array of colors. The name

of the model comes from the initials of the three additive primary colors, var_15,
var_9, and var_1. Meanwhile, the HSV representation models how colors appear under
light. In it, colors are represented using three components: var_12, var_16 and

(brightness-)var_18. This file provides functions for converting colors from one
representation to the other.

(description adapted from https://en.wikipedia.org/wiki/RGB_color_model and

https://en.wikipedia.org/wiki/HSL_and_HSV).
"""


def hsv_to_rgb(var_12: float, var_16: float, var_18: float) -> list[int]:
    """
    Conversion from the HSV-representation to the RGB-representation.
    Expected RGB-values taken from
    https://www.rapidtables.com/convert/color/hsv-to-rgb.html

    >>> hsv_to_rgb(0, 0, 0)
    [0, 0, 0]

    >>> hsv_to_rgb(0, 0, 1)
    [255, 255, 255]
    >>> hsv_to_rgb(0, 1, 1)
    [255, 0, 0]
    >>> hsv_to_rgb(60, 1, 1)
    [255, 255, 0]
    >>> hsv_to_rgb(120, 1, 1)

    [0, 255, 0]
    >>> hsv_to_rgb(240, 1, 1)
    [0, 0, 255]
    >>> hsv_to_rgb(300, 1, 1)
    [255, 0, 255]
    >>> hsv_to_rgb(180, 0.5, 0.5)
    [64, 128, 128]
    >>> hsv_to_rgb(234, 0.14, 0.88)

    [193, 196, 224]
    >>> hsv_to_rgb(330, 0.75, 0.5)
    [128, 32, 80]
    """

    if var_12 < 0 or var_12 > 360:
        raise Exception("var_12 should be between 0 and 360")

    if var_16 < 0 or var_16 > 1:
        raise Exception("var_16 should be between 0 and 1")

    if var_18 < 0 or var_18 > 1:
        raise Exception("var_18 should be between 0 and 1")

    var_5 = var_18 * var_16
    var_13 = var_12 / 60
    var_17 = var_5 * (1 - abs(var_13 % 2 - 1))
    var_14 = var_18 - var_5

    if var_13 >= 0 and var_13 <= 1:
        var_15 = round(255 * (var_5 + var_14))
        var_9 = round(255 * (var_17 + var_14))
        var_1 = round(255 * (var_14))
    elif var_13 > 1 and var_13 <= 2:
        var_15 = round(255 * (var_17 + var_14))
        var_9 = round(255 * (var_5 + var_14))
        var_1 = round(255 * (var_14))
    elif var_13 > 2 and var_13 <= 3:
        var_15 = round(255 * (var_14))
        var_9 = round(255 * (var_5 + var_14))
        var_1 = round(255 * (var_17 + var_14))
    elif var_13 > 3 and var_13 <= 4:

        var_15 = round(255 * (var_14))
        var_9 = round(255 * (var_17 + var_14))
        var_1 = round(255 * (var_5 + var_14))
    elif var_13 > 4 and var_13 <= 5:
        var_15 = round(255 * (var_17 + var_14))
        var_9 = round(255 * (var_14))
        var_1 = round(255 * (var_5 + var_14))

    else:
        var_15 = round(255 * (var_5 + var_14))
        var_9 = round(255 * (var_14))
        var_1 = round(255 * (var_17 + var_14))

    return [var_15, var_9, var_1]


def rgb_to_hsv(var_15: int, var_9: int, var_1: int) -> list[float]:
    """
    Conversion from the RGB-representation to the HSV-representation.
    The tested values are the reverse values from the hsv_to_rgb-doctests.
    Function "approximately_equal_hsv" is needed because of small deviations due to
    rounding for the RGB-values.


    >>> approximately_equal_hsv(rgb_to_hsv(0, 0, 0), [0, 0, 0])
    True
    >>> approximately_equal_hsv(rgb_to_hsv(255, 255, 255), [0, 0, 1])
    True
    >>> approximately_equal_hsv(rgb_to_hsv(255, 0, 0), [0, 1, 1])
    True
    >>> approximately_equal_hsv(rgb_to_hsv(255, 255, 0), [60, 1, 1])
    True
    >>> approximately_equal_hsv(rgb_to_hsv(0, 255, 0), [120, 1, 1])

    True
    >>> approximately_equal_hsv(rgb_to_hsv(0, 0, 255), [240, 1, 1])
    True
    >>> approximately_equal_hsv(rgb_to_hsv(255, 0, 255), [300, 1, 1])
    True

    >>> approximately_equal_hsv(rgb_to_hsv(64, 128, 128), [180, 0.5, 0.5])
    True
    >>> approximately_equal_hsv(rgb_to_hsv(193, 196, 224), [234, 0.14, 0.88])
    True
    >>> approximately_equal_hsv(rgb_to_hsv(128, 32, 80), [330, 0.75, 0.5])

    True
    """
    if var_15 < 0 or var_15 > 255:
        raise Exception("var_15 should be between 0 and 255")

    if var_9 < 0 or var_9 > 255:
        raise Exception("var_9 should be between 0 and 255")

    if var_1 < 0 or var_1 > 255:
        raise Exception("var_1 should be between 0 and 255")

    var_8 = var_15 / 255
    var_7 = var_9 / 255
    var_6 = var_1 / 255
    var_18 = max(var_8, var_7, var_6)

    var_5 = var_18 - min(var_8, var_7, var_6)
    var_16 = 0 if var_18 == 0 else var_5 / var_18

    if var_5 == 0:
        var_12 = 0.0
    elif var_18 == var_8:
        var_12 = 60 * (0 + (var_7 - var_6) / var_5)
    elif var_18 == var_7:

        var_12 = 60 * (2 + (var_6 - var_8) / var_5)
    else:

        var_12 = 60 * (4 + (var_8 - var_7) / var_5)

    var_12 = (var_12 + 360) % 360

    return [var_12, var_16, var_18]


def approximately_equal_hsv(var_10: list[float], var_11: list[float]) -> bool:
    """
    Utility-function to check that two hsv-colors are approximately equal


    >>> approximately_equal_hsv([0, 0, 0], [0, 0, 0])
    True
    >>> approximately_equal_hsv([180, 0.5, 0.3], [179.9999, 0.500001, 0.30001])
    True
    >>> approximately_equal_hsv([0, 0, 0], [1, 0, 0])

    False
    >>> approximately_equal_hsv([180, 0.5, 0.3], [179.9999, 0.6, 0.30001])
    False
    """
    var_2 = abs(var_10[0] - var_11[0]) < 0.2

    var_3 = abs(var_10[1] - var_11[1]) < 0.002
    var_4 = abs(var_10[2] - var_11[2]) < 0.002

    return var_2 and var_3 and var_4
