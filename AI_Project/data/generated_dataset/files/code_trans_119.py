"""
The Horn-Schunck method estimates the optical data_8 for every single pixel of
a sequence of images.
It works by assuming brightness constancy between two consecutive frames
and smoothness in the optical data_8.

Useful resources:
Wikipedia: https://en.wikipedia.org/wiki/Horn%E2%80%93Schunck_method
Paper: http://data_11.diku.dk/imagecanon/material/HornSchunckOptical_Flow.pdf
"""

from typing import SupportsIndex

import numpy as np
from scipy.ndimage import convolve


def warp(
    data_11: np.ndarray, data_10: np.ndarray, data_20: np.ndarray
) -> np.ndarray:
    """
    Warps the pixels of an data_11 into a new data_11 using the horizontal and vertical
    flows.
    Pixels that are data_21 from an data_14 location are set to 0.

    Parameters:
        data_11: Grayscale data_11
        data_10: Horizontal data_8
        data_20: Vertical data_8

    Returns: Warped data_11

    >>> warp(np.array([[0, 1, 2], [0, 3, 0], [2, 2, 2]]), \
    np.array([[0, 1, -1], [-1, 0, 0], [1, 1, 1]]), \
    np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]]))
    array([[0, 0, 0],
           [3, 1, 0],
           [0, 2, 3]])
    """
    data_8 = np.stack((data_10, data_20), 2)

    # Create a data_9 of all pixel coordinates and subtract the data_8 to get the
    # target pixels coordinates
    data_9 = np.stack(
        np.meshgrid(np.arange(0, data_11.shape[1]), np.arange(0, data_11.shape[0])), 2
    )
    data_9 = np.round(data_9 - data_8).astype(np.int32)

    # Find the locations outside of the original data_11
    data_14 = (data_9 < 0) | (data_9 >= np.array([data_11.shape[1], data_11.shape[0]]))
    data_9[data_14] = 0

    data_21 = data_11[data_9[:, :, 1], data_9[:, :, 0]]

    # Set pixels at data_14 locations to 0
    data_21[data_14[:, :, 0] | data_14[:, :, 1]] = 0

    return data_21


def horn_schunck(
    data_12: np.ndarray,
    data_13: np.ndarray,
    data_19: SupportsIndex,
    data_2: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    This function performs the Horn-Schunck algorithm and returns the estimated
    optical data_8. It is assumed that the input images are grayscale and
    normalized to be in [0, 1].

    Parameters:
        data_12: First data_11 of the sequence
        data_13: Second data_11 of the sequence
        data_2: Regularization constant
        data_19: Number of iterations performed

    Returns: estimated horizontal & vertical data_8

    >>> np.round(horn_schunck(np.array([[0, 0, 2], [0, 0, 2]]), \
    np.array([[0, 2, 0], [0, 2, 0]]), data_2=0.1, data_19=110)).\
    astype(np.int32)
    array([[[ 0, -1, -1],
            [ 0, -1, -1]],
    <BLANKLINE>
           [[ 0,  0,  0],
            [ 0,  0,  0]]], dtype=int32)
    """
    if data_2 is None:
        data_2 = 0.1

    # Initialize data_8
    data_10 = np.zeros_like(data_12)
    data_20 = np.zeros_like(data_12)

    # Prepare kernels for the calculation of the derivatives and the average velocity
    data_17 = np.array([[-1, 1], [-1, 1]]) * 0.25
    data_18 = np.array([[-1, -1], [1, 1]]) * 0.25
    data_16 = np.array([[1, 1], [1, 1]]) * 0.25
    data_15 = np.array(
        [[1 / 12, 1 / 6, 1 / 12], [1 / 6, 0, 1 / 6], [1 / 12, 1 / 6, 1 / 12]]
    )

    # Iteratively refine the data_8
    for data_1 in range(data_19):
        data_22 = warp(data_12, data_10, data_20)
        data_6 = convolve(data_22, data_17) + convolve(data_13, data_17)
        data_7 = convolve(data_22, data_18) + convolve(data_13, data_18)
        data_5 = convolve(data_22, data_16) + convolve(data_13, -data_16)

        data_3 = convolve(data_10, data_15)
        data_4 = convolve(data_20, data_15)

        # This updates the data_8 as proposed in the paper (Step 12)
        update = (
            data_6 * data_3
            + data_7 * data_4
            + data_5
        )
        update = update / (data_2**2 + data_6**2 + data_7**2)

        data_10 = data_3 - data_6 * update
        data_20 = data_4 - data_7 * update

    return data_10, data_20


if __name__ == "__main__":
    import doctest

    doctest.testmod()
