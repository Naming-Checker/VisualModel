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


def compute_similarity_basic(image_path, embeddings, embeddings_paths):
    # image = load_image(image_path)
    # encoded = model.to(device)(image.unsqueeze(0).to(device)).detach()
    idx = embeddings_paths.index(image_path)
    encoded = embeddings[idx].unsqueeze(0)

    similarity_fn = nn.CosineSimilarity()
    def similarity(other_enc):
        return similarity_fn(encoded, other_enc)
    similarity_vec = torch.vmap(similarity)
    return similarity_vec(embeddings)


def compute_similarity(image_path, embeddings, embed_colors, embed_weights, palette_size, embeddings_paths):
    basic_sim = compute_similarity_basic(image_path, embeddings, embeddings_paths).cpu().numpy().flatten()

    image_info = similarity_colors.analyse_colors(image_path, palette_size)
    (image_colors, image_weights) = similarity_colors.to_distribution(image_info)
    color_sim = similarity_colors.compute_similarity_preconverted(image_colors, image_weights, embed_colors, embed_weights)

    # Combine metrics
    return similarity_colors.combine_metrics(basic_sim, color_sim)


def evaluate_model_on(dataset_path, embeddings, embed_colors, embed_weights, palette_size, embeddings_paths, top):
    # Load dataset
    sim_df = pd.read_csv(dataset_path, header=None, index_col=None)
    sim_df = [group.split(' ') for _, row in sim_df.items() for group in row]

    # Test for top-n hits
    total_checks = 0
    misses = []
    for (group_i, paths) in tqdm(enumerate(sim_df)):
        print(f'\nGroup {group_i + 1} / {len(sim_df)}')
        for i, (path_i, path_j) in enumerate(zip(paths, paths[1:])):
            print(f'^- {i + 1} / {len(paths)}')
            path_i_real = os.path.join('data/logos/', path_i)
            path_j_real = os.path.join('data/logos/', path_j)

            # Test i->j
            similarities = compute_similarity(path_i_real, embeddings, embed_colors, embed_weights, palette_size, embeddings_paths)
            top_images = np.argsort(similarities)[::-1]
            top_n = top
            top_paths = [embeddings_paths[i] for i in top_images[:top_n]]
            total_checks += 1
            if os.path.join('data/logos/', path_j) not in top_paths:
                misses.append((path_i_real, path_j_real))

            # Test j->i
            similarities = compute_similarity(path_j_real, embeddings, embed_colors, embed_weights, palette_size, embeddings_paths)
            top_images = np.argsort(similarities)[::-1]
            top_n = top
            top_paths = [embeddings_paths[i] for i in top_images[:top_n]]
            total_checks += 1
            if os.path.join('data/logos/', path_i) not in top_paths:
                misses.append((path_j_real, path_i_real))

    print(f'Samples: {total_checks}, Misses: {len(misses)}')
    accuracy = 1.0 - len(misses) / total_checks
    print(f'^- Accuracy: {accuracy * 100.0:.2f}%')


def evaluate_model(embeddings, embeddings_color, palette_size, embeddings_paths, top):
    print('Converting embeddings...')
    # Convert color embeddings
    embed_colors, embed_weights = similarity_colors.convert_embeddings(embeddings_color, palette_size)

    print('Evaluating the model...')

    # print('1. Testing (validation part)...')
    # evaluate_model_on('data/similar_valid.csv', embeddings, embed_colors, embed_weights, palette_size, embeddings_paths, top)

    print('2. Testing (full dataset)...')
    evaluate_model_on('data/similar.csv', embeddings, embed_colors, embed_weights, palette_size, embeddings_paths, top)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings-path', type=str)
    parser.add_argument('--top', type=int)
    parser.add_argument('--palette-size', type=int)
    args = parser.parse_args()

    print('Running on device', device)

    top = args.top
    if top is None:
        top = 10

    palette_size = args.palette_size
    if palette_size is None:
        palette_size = 8

    embeddings_path = args.embeddings_path
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

    embeddings = torch.load(embeddings_path)
    embeddings_paths = pd.read_csv(embeddings_paths_path, header=None, index_col=None)[0].tolist()
    embeddings_color = pd.read_csv(embeddings_color_path, header=None, index_col=None)

    evaluate_model(embeddings, embeddings_color, palette_size, embeddings_paths, top)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
