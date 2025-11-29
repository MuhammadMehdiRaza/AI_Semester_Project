"""
Principal Component Analysis (PCA) is a dimensionality reduction technique
used in machine learning. It transforms high-dimensional temp_2 into a lower-dimensional
representation while retaining as much variance as possible.

This implementation follows best practices, including:
- Standardizing the dataset.
- Computing principal components using Singular Value Decomposition (SVD).
- Returning transformed temp_2 and explained variance ratio.
"""

import doctest

import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def do_2() -> tuple[np.ndarray, np.ndarray]:
    """
    Collects the dataset (Iris dataset) and returns feature matrix and target values.

    :return: Tuple containing feature matrix (X) and target labels (y)

    Example:
    >>> X, y = do_2()
    >>> X.shape
    (150, 4)
    >>> y.shape
    (150,)
    """
    temp_2 = load_iris()
    return np.array(temp_2.temp_2), np.array(temp_2.target)


def do_1(temp_3: np.ndarray, temp_5: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies Principal Component Analysis (PCA) to reduce dimensionality.

    :param temp_3: Original dataset (features)
    :param temp_5: Number of principal components to retain
    :return: Tuple containing transformed dataset and explained variance ratio

    Example:
    >>> X, _ = do_2()
    >>> transformed_X, variance = do_1(X, 2)
    >>> transformed_X.shape
    (150, 2)
    >>> len(variance) == 2
    True
    """
    # Standardizing the dataset
    temp_8 = StandardScaler()
    temp_4 = temp_8.fit_transform(temp_3)

    # Applying PCA
    temp_6 = PCA(temp_5=temp_5)
    temp_7 = temp_6.fit_transform(temp_4)

    return temp_7, temp_6.explained_variance_ratio_


def do_3() -> None:
    """
    Driver function to execute PCA and display results.
    """
    temp_3, temp_1 = do_2()

    # Number of principal components to retain
    temp_5 = 2

    # Apply PCA
    temp_9, temp_10 = do_1(temp_3, temp_5)

    print("Transformed Dataset (First 5 rows):")
    print(temp_9[:5])

    print("\nExplained Variance Ratio:")
    print(temp_10)


if __name__ == "__main__":
    doctest.testmod()
    do_3()
