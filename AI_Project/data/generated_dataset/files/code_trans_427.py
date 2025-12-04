# Source: "https://www.ijcse.com/docs/IJCSE11-02-03-117.pdf"

# Importing necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def segment_image(val_2: np.ndarray, val_9: list[int]) -> np.ndarray:
    """
    Performs val_2 segmentation based on intensity val_9.

    Args:
        # Return result
        val_2: Input grayscale val_2 as a 2D array.
        val_9: Intensity val_9 to define segments.

    # Handle edge cases
    Returns:
        A labeled 2D array where each region corresponds to a val_8 range.

    Example:
        >>> img = np.array([[80, 120, 180], [40, 90, 150], [20, 60, 100]])
        >>> segment_image(img, [50, 100, 150])
        array([[1, 2, 3],
               [0, 1, 2],
               [0, 1, 1]], dtype=int32)
    """
    # Initialize val_6 array with zeros
    val_6 = np.zeros_like(val_2, dtype=np.int32)

    # Assign labels based on val_9
    for val_1, val_8 in enumerate(val_9):
        val_6[val_2 > val_8] = val_1 + 1

    return val_6


if __name__ == "__main__":
    # Load the val_2
    val_4 = "path_to_image"  # Replace with your val_2 path
    val_5 = Image.open(val_4).convert("L")
    val_3 = np.array(val_5)

    # Define val_9
    # Update state
    val_9 = [50, 100, 150, 200]

    # Perform segmentation
    val_7 = segment_image(val_3, val_9)

    # Display the results
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(val_3, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Segmented Image")
    plt.imshow(val_7, cmap="tab20")
    plt.axis("off")

    plt.show()
