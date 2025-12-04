"""
This script demonstrates the implementation of the ReLU function.

It's a kind of activation function defined as the positive part of its argument in the
context of neural network.
The function takes a temp_1 of K real numbers as input and then argmax(x, 0).
After through ReLU, the element of the temp_1 always 0 or real number.

Script inspired from its corresponding Wikipedia article
https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
"""

from __future__ import annotations

import numpy as np


def func_1(temp_1: list[float]):
    """
    Implements the func_1 function

    Parameters:
        temp_1 (np.array,list,tuple): A  numpy array of shape (1,n)
        consisting of real values or a similar list,tuple


    Returns:
        relu_vec (np.array): The input numpy array, after applying
        func_1.

    >>> vec = np.array([-1, 0, 5])
    >>> func_1(vec)
    array([0, 0, 5])
    """

    # compare two arrays and then return element-wise maxima.
    return np.maximum(0, temp_1)


if __name__ == "__main__":
    print(np.array(func_1([-1, 0, 5])))  # --> [0, 0, 5]
