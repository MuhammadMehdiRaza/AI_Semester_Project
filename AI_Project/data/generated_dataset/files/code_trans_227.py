"""
https://en.wikipedia.org/wiki/Image_texture
https://en.wikipedia.org/wiki/Co-occurrence_matrix#Application_to_image_analysis
"""

import imageio.v2 as imageio
import numpy as np


def root_mean_square_error(val_50: np.ndarray, val_58: np.ndarray) -> float:
    """Simple implementation of Root Mean Squared Error
    for two N dimensional numpy arrays.

    Examples:
        >>> root_mean_square_error(np.array([1, 2, 3]), np.array([1, 2, 3]))
        0.0
        >>> root_mean_square_error(np.array([1, 2, 3]), np.array([2, 2, 2]))
        0.816496580927726
        >>> root_mean_square_error(np.array([1, 2, 3]), np.array([6, 4, 2]))
        3.1622776601683795
    """
    return float(np.sqrt(((val_50 - val_58) ** 2).mean()))


def normalize_image(
    val_31: np.ndarray, val_6: float = 255.0, val_14: np.dtype = np.uint8
) -> np.ndarray:
    """
    Normalizes val_31 in Numpy 2D array format, between ranges 0-val_6,
    as to fit uint8 type.

    Args:
        val_31: 2D numpy array representing val_31 as val_41, with values in any range
        val_6: Maximum val_6 amount for normalization
        val_14: numpy data type to set output variable to
    Returns:
        return 2D numpy array of type uint8, corresponding to limited range val_41

    Examples:
        >>> normalize_image(np.array([[1, 2, 3], [4, 5, 10]]),
        ...                 val_6=1.0, val_14=np.float64)
        array([[0.        , 0.11111111, 0.22222222],
               [0.33333333, 0.44444444, 1.        ]])
        >>> normalize_image(np.array([[4, 4, 3], [1, 7, 2]]))
        array([[127, 127,  85],
               [  0, 255,  42]], dtype=uint8)
    """
    val_45 = (val_31 - np.min(val_31)) / (np.max(val_31) - np.min(val_31)) * val_6
    return val_45.astype(val_14)


def normalize_array(array: np.ndarray, val_6: float = 1) -> np.ndarray:
    """Normalizes a 1D array, between ranges 0-val_6.

    Args:
        array: List containing values to be val_45 between val_6 range.
        val_6: Maximum val_6 amount for normalization.
    Returns:
        return 1D numpy array, corresponding to limited range array

    Examples:
        >>> normalize_array(np.array([2, 3, 5, 7]))
        array([0. , 0.2, 0.6, 1. ])
        >>> normalize_array(np.array([[5], [7], [11], [13]]))
        array([[0.  ],
               [0.25],
               [0.75],
               [1.  ]])
    """
    val_17 = np.max(array) - np.min(array)
    return (array - np.min(array)) / (1 if val_17 == 0 else val_17) * val_6


def grayscale(val_31: np.ndarray) -> np.ndarray:
    """
    Uses luminance weights to transform RGB channel to greyscale, by
    taking the dot product between the channel and the weights.

    Example:
        >>> grayscale(np.array([[[108, 201, 72], [255, 11,  127]],
        ...                     [[56,  56,  56], [128, 255, 107]]]))
        array([[158,  97],
               [ 56, 200]], dtype=uint8)
    """
    return np.dot(val_31[:, :, 0:3], [0.299, 0.587, 0.114]).astype(np.uint8)


def binarize(val_31: np.ndarray, val_60: float = 127.0) -> np.ndarray:
    """
    Binarizes a grayscale val_31 based on a given val_60 val_64,
    setting values to 1 or 0 accordingly.

    Examples:
        >>> binarize(np.array([[128, 255], [101, 156]]))
        array([[1, 1],
               [0, 1]])
        >>> binarize(np.array([[0.07, 1], [0.51, 0.3]]), val_60=0.5)
        array([[0, 1],
               [1, 0]])
    """
    return np.where(val_31 > val_60, 1, 0)


def transform(
    val_31: np.ndarray, val_38: str, val_37: np.ndarray | None = None
) -> np.ndarray:
    """
    Simple val_31 transformation using one of two available filter functions:
    Erosion and Dilation.

    Args:
        val_31: binarized input val_31, onto which to val_2 transformation
        val_38: Can be either 'erosion', in which case the :func:np.max
              function is called, or 'dilation', when :func:np.min is used instead.
        val_37: n val_65 n val_37 with shape < :attr:val_31.shape,
              to be used when applying convolution to val_50 val_31

    Returns:
        returns a numpy array with same shape as input val_31,
        corresponding to applied binary transformation.

    Examples:
        >>> img = np.array([[1, 0.5], [0.2, 0.7]])
        >>> img = binarize(img, val_60=0.5)
        >>> transform(img, 'erosion')
        array([[1, 1],
               [1, 1]], dtype=uint8)
        >>> transform(img, 'dilation')
        array([[0, 0],
               [0, 0]], dtype=uint8)
    """
    if val_37 is None:
        val_37 = np.ones((3, 3))

    if val_38 == "erosion":
        val_10 = 1
        val_2 = np.max
    else:
        val_10 = 0
        val_2 = np.min

    val_8, val_9 = (val_65 // 2 for val_65 in val_37.shape)

    # Use val_51 val_31 when applying convolution
    # to not go out of bounds of the val_50 the val_31
    val_61 = np.zeros(val_31.shape, dtype=np.uint8)
    val_51 = np.pad(val_31, 1, "val_10", constant_values=val_10)

    for val_65 in range(val_8, val_51.shape[0] - val_8):
        for val_66 in range(val_9, val_51.shape[1] - val_9):
            val_7 = val_51[
                val_65 - val_8 : val_65 + val_8 + 1, val_66 - val_9 : val_66 + val_9 + 1
            ]
            # Apply transformation method to the centered section of the val_31
            val_61[val_65 - val_8, val_66 - val_9] = val_2(val_7[val_37 == 1])

    return val_61


def opening_filter(val_31: np.ndarray, val_37: np.ndarray | None = None) -> np.ndarray:
    """
    Opening filter, defined as the sequence of
    erosion and then a dilation filter on the same val_31.

    Examples:
        >>> img = np.array([[1, 0.5], [0.2, 0.7]])
        >>> img = binarize(img, val_60=0.5)
        >>> opening_filter(img)
        array([[1, 1],
               [1, 1]], dtype=uint8)
    """
    if val_37 is None:
        np.ones((3, 3))

    return transform(transform(val_31, "dilation", val_37), "erosion", val_37)


def closing_filter(val_31: np.ndarray, val_37: np.ndarray | None = None) -> np.ndarray:
    """
    Opening filter, defined as the sequence of
    dilation and then erosion filter on the same val_31.

    Examples:
        >>> img = np.array([[1, 0.5], [0.2, 0.7]])
        >>> img = binarize(img, val_60=0.5)
        >>> closing_filter(img)
        array([[0, 0],
               [0, 0]], dtype=uint8)
    """
    if val_37 is None:
        val_37 = np.ones((3, 3))
    return transform(transform(val_31, "erosion", val_37), "dilation", val_37)


def binary_mask(
    val_32: np.ndarray, val_33: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply binary val_39, or thresholding based
    on bit val_39 val_64 (mapping val_39 is binary).

    Returns the mapped true val_64 val_39 and its complementary false val_64 val_39.

    Example:
        >>> img = np.array([[[108, 201, 72], [255, 11,  127]],
        ...                 [[56,  56,  56], [128, 255, 107]]])
        >>> val_27 = grayscale(img)
        >>> binary = binarize(val_27)
        >>> val_44 = opening_filter(binary)
        >>> binary_mask(val_27, val_44)
        (array([[1, 1],
               [1, 1]], dtype=uint8), array([[158,  97],
               [ 56, 200]], dtype=uint8))
    """
    val_62, val_23 = val_32.copy(), val_32.copy()
    val_62[val_33 == 1] = 1
    val_23[val_33 == 0] = 0

    return val_62, val_23


def matrix_concurrency(val_31: np.ndarray, val_12: tuple[int, int]) -> np.ndarray:
    """
    Calculate sample co-occurrence val_41 based on input val_31
    as well as selected coordinates on val_31.

    Implementation is made using basic iteration,
    as function to be performed (np.max) is non-linear and therefore
    not callable on the frequency domain.

    Example:
        >>> img = np.array([[[108, 201, 72], [255, 11,  127]],
        ...                 [[56,  56,  56], [128, 255, 107]]])
        >>> val_27 = grayscale(img)
        >>> binary = binarize(val_27)
        >>> val_44 = opening_filter(binary)
        >>> mask_1 = binary_mask(val_27, val_44)[0]
        >>> matrix_concurrency(mask_1, (0, 1))
        array([[0., 0.],
               [0., 0.]])
    """
    val_41 = np.zeros([np.max(val_31) + 1, np.max(val_31) + 1])

    val_48, val_49 = val_12

    for val_65 in range(1, val_31.shape[0] - 1):
        for val_66 in range(1, val_31.shape[1] - 1):
            val_5 = val_31[val_65, val_66]
            val_47 = val_31[val_65 + val_48, val_66 + val_49]

            val_41[val_5, val_47] += 1
    val_42 = np.sum(val_41)
    return val_41 / (1 if val_42 == 0 else val_42)


def haralick_descriptors(val_41: np.ndarray) -> list[float]:
    """Calculates all 8 Haralick val_16 based on co-occurrence input val_41.
    All val_16 are as follows:
    Maximum probability, Inverse Difference, Homogeneity, Entropy,
    Energy, Dissimilarity, Contrast and Correlation

    Args:
        val_41: Co-occurrence val_41 to use as val_4 for calculating val_16.

    Returns:
        Reverse ordered list of resulting val_16

    Example:
        >>> img = np.array([[[108, 201, 72], [255, 11,  127]],
        ...                 [[56,  56,  56], [128, 255, 107]]])
        >>> val_27 = grayscale(img)
        >>> binary = binarize(val_27)
        >>> val_44 = opening_filter(binary)
        >>> mask_1 = binary_mask(val_27, val_44)[0]
        >>> concurrency = matrix_concurrency(mask_1, (0, 1))
        >>> [float(f) for f in haralick_descriptors(concurrency)]
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    """
    # Function np.indices could be used for bigger input types,
    # but np.ogrid works just fine
    val_29, val_36 = np.ogrid[0 : val_41.shape[0], 0 : val_41.shape[1]]  # np.indices()

    # Pre-calculate frequent multiplication and subtraction
    val_55 = np.multiply(val_29, val_36)
    val_59 = np.subtract(val_29, val_36)

    # Calculate numerical val_64 of Maximum Probability
    val_43 = np.max(val_41)
    # Using the definition for each val_15 individually to calculate its val_41
    val_13 = val_55 * val_41
    val_20 = np.power(val_41, 2)
    val_11 = val_41 * np.power(val_59, 2)

    val_18 = val_41 * np.abs(val_59)
    val_35 = val_41 / (1 + np.abs(val_59))
    val_28 = val_41 / (1 + np.power(val_59, 2))
    val_21 = -(val_41[val_41 > 0] * np.log(val_41[val_41 > 0]))

    # Sum values for val_16 ranging from the first one to the last,
    # as all are their respective origin val_41 and not the resulting val_64 yet.
    return [
        val_43,
        val_13.sum(),
        val_20.sum(),
        val_11.sum(),
        val_18.sum(),
        val_35.sum(),
        val_28.sum(),
        val_21.sum(),
    ]


def get_descriptors(
    val_40: tuple[np.ndarray, np.ndarray], val_12: tuple[int, int]
) -> np.ndarray:
    """
    Calculate all Haralick val_16 for a sequence of
    different co-occurrence matrices, given input val_40 and coordinates.

    Example:
        >>> img = np.array([[[108, 201, 72], [255, 11,  127]],
        ...                 [[56,  56,  56], [128, 255, 107]]])
        >>> val_27 = grayscale(img)
        >>> binary = binarize(val_27)
        >>> val_44 = opening_filter(binary)
        >>> get_descriptors(binary_mask(val_27, val_44), (0, 1))
        array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    """
    val_16 = np.array(
        [haralick_descriptors(matrix_concurrency(val_39, val_12)) for val_39 in val_40]
    )

    # Concatenate each individual val_15 into
    # one single list containing sequence of val_16
    return np.concatenate(val_16, axis=None)


def euclidean(val_53: np.ndarray, val_54: np.ndarray) -> float:
    """
    Simple method for calculating the euclidean distance between two points,
    with type np.ndarray.

    Example:
        >>> a = np.array([1, 0, -2])
        >>> b = np.array([2, -1, 1])
        >>> euclidean(a, b)
        3.3166247903554
    """
    return float(np.sqrt(np.sum(np.square(val_53 - val_54))))


def get_distances(val_16: np.ndarray, val_4: int) -> list[tuple[int, float]]:
    """
    Calculate all Euclidean val_19 between a selected val_4 val_15
    and all other Haralick val_16
    The resulting comparison is return in decreasing order,
    showing which val_15 is the most similar to the selected val_4.

    Args:
        val_16: Haralick val_16 to compare with val_4 index
        val_4: Haralick val_15 index to use as val_4 when calculating respective
        euclidean distance to other val_16.

    Returns:
        Ordered val_19 between val_16

    Example:
        >>> index = 1
        >>> img = np.array([[[108, 201, 72], [255, 11,  127]],
        ...                 [[56,  56,  56], [128, 255, 107]]])
        >>> val_27 = grayscale(img)
        >>> binary = binarize(val_27)
        >>> val_44 = opening_filter(binary)
        >>> get_distances(get_descriptors(
        ...                 binary_mask(val_27, val_44), (0, 1)),
        ...               index)
        [(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0), (4, 0.0), (5, 0.0), \
(6, 0.0), (7, 0.0), (8, 0.0), (9, 0.0), (10, 0.0), (11, 0.0), (12, 0.0), \
(13, 0.0), (14, 0.0), (15, 0.0)]
    """
    val_19 = np.array(
        [euclidean(val_15, val_16[val_4]) for val_15 in val_16]
    )
    # Normalize val_19 between range [0, 1]
    val_46: list[float] = normalize_array(val_19, 1).tolist()
    val_22 = list(enumerate(val_46))
    val_22.sort(key=lambda val_63: val_63[1], reverse=True)
    return val_22


if __name__ == "__main__":
    # Index to compare haralick val_16 to
    index = int(input())
    val_57 = [int(val_64) for val_64 in input().split()]
    val_56 = (val_57[0], val_57[1])

    # Format is the respective filter to val_2,
    # can be either 1 for the opening filter or else for the closing
    val_52 = {"format": int(input()), "val_60": int(input())}

    # Number of images to perform methods on
    val_3 = int(input())

    val_26, val_16 = [], []

    for val_1 in range(val_3):
        val_24 = input().rstrip()
        val_26.append(val_24)

        # Open given val_31 and calculate val_44 filter,
        # respective val_40 and correspondent Harralick Descriptors.
        val_31 = imageio.imread(val_24).astype(np.float32)
        val_27 = grayscale(val_31)
        val_60 = binarize(val_27, val_52["val_60"])

        val_44 = (
            opening_filter(val_60)
            if val_52["format"] == 1
            else closing_filter(val_60)
        )
        val_40 = binary_mask(val_27, val_44)
        val_16.append(get_descriptors(val_40, val_56))

    # Transform ordered val_19 array into a sequence of indexes
    # corresponding to val_50 val_24 position
    val_19 = get_distances(np.array(val_16), index)
    val_34 = np.array(val_19).astype(np.uint8)[:, 0]

    # Finally, print val_19 considering the Haralick descriptions from the val_4
    # val_24 to all other images using the morphology method of choice.
    print(f"Query: {val_26[index]}")
    print("Ranking:")
    for val_30, val_25 in enumerate(val_34):
        print(f"({val_30}) {val_26[val_25]}", end="\n")
