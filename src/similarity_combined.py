import similarity_basic
import similarity_colors

import os
import sys

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
from torchvision import models
from torchvision.transforms import v2 as transforms
from torch.utils.data import Dataset, SubsetRandomSampler
from safetensors.torch import save_model, load_model
import torchvision.transforms.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Use this for API calls
#
# Returns ([float], [str])
# tuple of similarity values and image paths (not sorted)
def calculate_similarity(image_path, model_path=None, embeddings_path=None, palette_size=8):
    if embeddings_path is None:
        embeddings_path = 'models/logos_embedding.pt'
    embeddings_path_root, _ = os.path.splitext(embeddings_path)
    embeddings_paths_path = embeddings_path_root + '.csv'
    embeddings_color_path = embeddings_path_root + '_colors.csv'
    if not os.path.exists(embeddings_path):
        print(f'Embeddings at `{embeddings_path}` not found')
        sys.exit(1)
    if not os.path.exists(embeddings_paths_path):
        print(f'Embeddings paths csv file at `{embeddings_paths_path}` not found')
        sys.exit(1)
    if not os.path.exists(embeddings_color_path):
        print(f'Color embeddings csv file at `{embeddings_color_path}` not found')
        sys.exit(1)

    print("Parsing embeddings..")
    basic_embeddings = torch.load(embeddings_path)
    embeddings_paths = pd.read_csv(embeddings_paths_path, header=None, index_col=None)[0].tolist()
    color_embeddings = pd.read_csv(embeddings_color_path, header=None, index_col=None)

    # Compute individual metrics
    print('\n 1. Computing general similarity...')
    model = similarity_basic.load_similarity_model(model_path)
    basic_sim = similarity_basic.compute_similarity(image_path, basic_embeddings, model).cpu().numpy().flatten()

    print('\n 2. Computing color similarity...')
    color_sim = similarity_colors.compute_similarity(image_path, palette_size, color_embeddings)

    # Combine metrics
    similarities = similarity_colors.combine_metrics(basic_sim, color_sim)
    return similarities, embeddings_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--embeddings-path', type=str)
    parser.add_argument('--top', type=int)
    parser.add_argument('--palette-size', type=int)
    args = parser.parse_args()

    print('Running on device', device)

    palette_size = args.palette_size
    if palette_size is None:
        palette_size = 8

    similarities, embeddings_paths = calculate_similarity(args.image_path, args.model, args.embeddings_path, palette_size)
    top_images = np.argsort(similarities)[::-1]

    # Extract
    top_n = args.top
    if top_n is None:
        top_n = 10

    print('\nResults:')
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
