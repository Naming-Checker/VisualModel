import argparse
from color_analysis_tool import ImageAnalyzer, ImageInfo
#from matplotlib.colors import rgb_to_hsv
from tqdm import tqdm
import numpy as np
from scipy.stats import wasserstein_distance_nd


def analyse_colors(image_path, palette_size):
    analyzer = ImageAnalyzer()
    return analyzer.analyze_image(image_path, max_colors=palette_size)


# For some reason, the function from matplotlib doesn't work
def rgb_to_hsv(arr):
    """
    Convert an array of float RGB values (in the range [0, 1]) to HSV values.

    Parameters
    ----------
    arr : (..., 3) array-like
       All values must be in the range [0, 1]

    Returns
    -------
    (..., 3) `~numpy.ndarray`
       Colors converted to HSV values in range [0, 1]
    """
    arr = np.asarray(arr)

    # check length of the last dimension, should be _some_ sort of rgb
    if arr.shape[-1] != 3:
        raise ValueError("Last dimension of input array must be 3; "
                         f"shape {arr.shape} was found.")

    in_shape = arr.shape
    arr = np.array(
        arr,
        dtype=np.promote_types(arr.dtype, np.float32),  # Don't work on ints.
        ndmin=2,  # In case input was 1D.
    )

    out = np.zeros_like(arr)
    arr_max = arr.max(-1)
    # Check if input is in the expected range
    if np.any(arr_max > 1):
        raise ValueError(
            "Input array must be in the range [0, 1]. "
            f"Found a maximum value of {arr_max.max()}"
        )

    if arr.min() < 0:
        raise ValueError(
            "Input array must be in the range [0, 1]. "
            f"Found a minimum value of {arr.min()}"
        )

    ipos = arr_max > 0
    delta = np.ptp(arr, -1)
    s = np.zeros_like(delta)
    s[ipos] = delta[ipos] / arr_max[ipos]
    ipos = delta > 0
    # red is max
    idx = (arr[..., 0] == arr_max) & ipos
    out[idx, 0] = (arr[idx, 1] - arr[idx, 2]) / delta[idx]
    # green is max
    idx = (arr[..., 1] == arr_max) & ipos
    out[idx, 0] = 2. + (arr[idx, 2] - arr[idx, 0]) / delta[idx]
    # blue is max
    idx = (arr[..., 2] == arr_max) & ipos
    out[idx, 0] = 4. + (arr[idx, 0] - arr[idx, 1]) / delta[idx]

    out[..., 0] = (out[..., 0] / 6.0) % 1.0
    out[..., 1] = s
    out[..., 2] = arr_max

    return out.reshape(in_shape)


# Convert analysis result to a tuple of values/weights
def to_distribution(info: ImageInfo):
    values = []
    weights = []
    for color in info.colors:
        (r,g,b) = color.rgb
        values.append(rgb_to_hsv([r / 255.0, g / 255.0, b / 255.0]))
        weights.append(color.frequency)
    return (values, weights)


# TODO: similarity instead of distance
def compute_distance(image_path_a, image_path_b, palette_size):
    info_a = analyse_colors(image_path_a, palette_size)
    (colors_a, weights_a) = to_distribution(info_a)
    info_b = analyse_colors(image_path_b, palette_size)
    (colors_b, weights_b) = to_distribution(info_b)
    distance = wasserstein_distance_nd(colors_a, colors_b, weights_a, weights_b)
    return (info_a, info_b, distance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str)
    parser.add_argument('--palette-size', type=int)
    parser.add_argument('--compare', type=str)
    args = parser.parse_args()

    palette_size = args.palette_size
    if palette_size is None:
        palette_size = 8

    if args.compare is not None:
        # Compare two images
        (info_a, info_b, distance) = compute_distance(args.image_path, args.compare, palette_size)
        print(f"Image A {args.image_path}:")
        for color in info_a.colors:
            print(f"- RGB: {color.rgb}, HEX: {color.hex}, Frequency: {color.frequency}%")
        print(f"\nImage B {args.compare}:")
        for color in info_b.colors:
            print(f"- RGB: {color.rgb}, HEX: {color.hex}, Frequency: {color.frequency}%")
        print(f"\nDistance: {distance}")
        print(f"Similarity: {1.0 / (1.0 + distance) * 100.0 : .2f}%")
    else:
        # Analyse single image
        info = analyse_colors(args.image_path, palette_size)
        for color in info.colors:
            print(f"RGB: {color.rgb}, HEX: {color.hex}, Frequency: {color.frequency}%")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
