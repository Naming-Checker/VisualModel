import os

import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import multiprocessing as mp

from color_analysis_tool import ImageAnalyzer, ImageInfo


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
        hsv = rgb_to_hsv([r / 255.0, g / 255.0, b / 255.0])
        values.append(hsv.tolist())
        weights.append(color.frequency)
    return (values, weights)


def get_embeddings(args):
    image_path, palette_size = args
    info = analyse_colors(image_path, palette_size)
    (colors, weights) = to_distribution(info)
    encoding = (colors, weights)
    return [encoding]


def generate_embeddings(image_paths, palette_size):
    print("Generating color embeddings...")
    # embeddings = []
    # for image_path in tqdm(image_paths):
    #     info = analyse_colors(image_path, palette_size)
    #     (colors, weights) = to_distribution(info)
    #     encoding = (colors, weights)
    #     embeddings.append(encoding)
    embeddings = []
    with mp.Pool(mp.cpu_count()) as pool:
        it = pool.imap(get_embeddings, [(image_paths[i], palette_size) for i in range(len(image_paths))])
        embeddings = list(tqdm(it, total=len(image_paths)))
    return embeddings


def load_paths(embedding_paths):
    df = pd.read_csv(embedding_paths, header=None, index_col=None)
    return df.to_numpy().flatten()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('embedding_paths', type=str)
    parser.add_argument('--palette-size', type=int)
    args = parser.parse_args()

    palette_size = args.palette_size
    if palette_size is None:
        palette_size = 8

    paths_path = args.embedding_paths
    if paths_path is None:
        paths_path = 'models/logos_embedding.pt'
    target_path, _ = os.path.splitext(paths_path)
    target_path = target_path + '_colors.csv'
    print(f'Saving embeddings to `{target_path}`')

    paths = load_paths(paths_path)
    embeddings = generate_embeddings(paths, palette_size)
    pd.DataFrame(embeddings).to_csv(target_path, header=None, index=False)
    print('Embeddings saved')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
