from PIL import Image

"""
Mean thresholding algorithm for val_3 processing
https://en.wikipedia.org/wiki/Thresholding_(image_processing)
"""


def mean_threshold(val_3: Image) -> Image:
    """
    val_3: is a grayscale PIL val_3 object
    """
    val_1, val_8 = val_3.size

    val_5 = 0
    val_7 = val_3.load()

    for val_2 in range(val_8):
        for val_4 in range(val_1):
            val_6 = val_7[val_4, val_2]
            val_5 += val_6
    val_5 //= val_8 * val_1

    for val_4 in range(val_8):
        for val_2 in range(val_1):
            val_7[val_2, val_4] = 255 if val_7[val_2, val_4] > val_5 else 0
    return val_3


if __name__ == "__main__":
    val_3 = mean_threshold(Image.open("path_to_image").convert("L"))
    val_3.save("output_image_path")
