import os
import sys
import ast
import multiprocessing as mp

import csv
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.io import read_image
from torchvision.transforms import v2 as transforms
from torch.utils.data import Dataset, SubsetRandomSampler
import torchvision.transforms.functional as F

from color_analysis_tool import ImageAnalyzer, ImageInfo
from scipy.stats import wasserstein_distance_nd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        values.append([r / 255.0, g / 255.0, b / 255.0])
        weights.append(color.frequency)
    return (values, weights)


def similarity(args):
    colors, weights, other_colors, other_weights = args
    distance = wasserstein_distance_nd(colors, other_colors, weights, other_weights)
    return 1.0 / (1.0 + distance)


def convert_embeddings(embeddings, palette_size):
    embed_colors = []
    embed_weights = []
    for _, row in tqdm(embeddings.iterrows(), total=len(embeddings)):
        colors, weights = ast.literal_eval(row[0])
        missing = (palette_size - len(colors))
        embed_colors.append(colors + [[0.0,0.0,0.0]] * missing)
        embed_weights.append(weights + [0.0] * missing)
        assert len(embed_colors[-1]) == palette_size
        assert len(embed_weights[-1]) == palette_size
    return embed_colors, embed_weights


# Use this with preconverted embeddings and color analysis
def compute_similarity_preconverted(image_colors, image_weights, embed_colors, embed_weights, tqdm_on=True):
    with mp.Pool(mp.cpu_count()) as pool:
        tasks = [
            (image_colors, image_weights, other_c, other_w) 
            for other_c, other_w in zip(embed_colors, embed_weights)
        ]
        it = pool.imap(similarity, tasks, chunksize=10)
        if tqdm_on:
            embeddings = list(tqdm(it, total=len(embed_colors)))
        else:
            embeddings = list(it)
    return embeddings


def combine_metrics(basic_sim, color_sim):
    median_color_sim = np.median(color_sim)

    assert len(basic_sim) == len(color_sim)
    k = 7 # the factor of the color curve, bigger number - less impact at lower values
    m = 0.1 * (1.0 - median_color_sim) # extra multiplier to reduce max impact of the color similarity
    similarities = []
    for i in range(len(basic_sim)):
        b = basic_sim[i]
        c = color_sim[i]
        f = c ** k * m
        value = b * (1.0 - f) + f * c
        similarities.append(value)
    return similarities


# Use this with preloaded embeddings (they will be converted to the internal format).
def compute_similarity(image_path, palette_size, embeddings):
    info = analyse_colors(image_path, palette_size)
    (image_colors, image_weights) = to_distribution(info)

    # Print image analysis
    for color in info.colors:
        print(f"RGB: {color.rgb}, HEX: {color.hex}, Frequency: {color.frequency}%")

    print("Converting embeddings...")
    embed_colors, embed_weights = convert_embeddings(embeddings, palette_size)

    print("Calculating similarity...")
    return compute_similarity_preconverted(image_colors, image_weights, embed_colors, embed_weights)


# Use this for API calls
#
# Returns ([float], [str])
# tuple of similarity values and image paths (not sorted)
def calculate_similarity(image_path, palette_size, embeddings_path=None):
    if embeddings_path is None:
        embeddings_path = 'models/logos_embedding_colors.csv'
    embeddings_path_root, _ = os.path.splitext(embeddings_path)
    embeddings_path_root = embeddings_path_root.removesuffix('_colors')
    embeddings_paths_path = embeddings_path_root + '.csv'
    if not os.path.exists(embeddings_path):
        print(f'Embeddings at `{embeddings_path}` not found')
        sys.exit(1)
    print(embeddings_paths_path)
    if not os.path.exists(embeddings_paths_path):
        print(f'Embeddings paths csv file at `{embeddings_paths_path}` not found')
        sys.exit(1)

    print("Parsing embeddings..")
    embeddings = pd.read_csv(embeddings_path, header=None, index_col=None)
    embeddings_paths = pd.read_csv(embeddings_paths_path, header=None, index_col=None)[0].tolist()

    # Compute
    similarities = compute_similarity(image_path, palette_size, embeddings)
    return similarities, embeddings_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str)
    parser.add_argument('--embeddings-path', type=str)
    parser.add_argument('--palette-size', type=int)
    parser.add_argument('--top', type=int)
    args = parser.parse_args()

    palette_size = args.palette_size
    if palette_size is None:
        palette_size = 8

    similarities, embeddings_paths = calculate_similarity(args.image_path, palette_size, args.embeddings_path)
    top_images = np.argsort(similarities)[::-1]

    # Extract
    top_n = args.top
    if top_n is None:
        top_n = 10

    for i in range(top_n):
        idx = top_images[i]
        sim = similarities[idx] * 100
        path = embeddings_paths[idx]
        print(f"{i}. {sim:.2f}% - '{path}'")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
