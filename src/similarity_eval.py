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


def compute_similarity(image_path, embeddings, embeddings_paths):
    # image = load_image(image_path)
    # encoded = model.to(device)(image.unsqueeze(0).to(device)).detach()
    idx = embeddings_paths.index(image_path)
    encoded = embeddings[idx].unsqueeze(0)

    similarity_fn = nn.CosineSimilarity()
    def similarity(other_enc):
        return similarity_fn(encoded, other_enc)
    similarity_vec = torch.vmap(similarity)
    return similarity_vec(embeddings)


def evaluate_model_on(dataset_path, embeddings, embeddings_paths, top):
    # Load dataset
    sim_df = pd.read_csv(dataset_path, header=None, index_col=None)
    sim_df = [group.split(' ') for _, row in sim_df.items() for group in row]

    # Test for top-n hits
    total_checks = 0
    misses = []
    for paths in tqdm(sim_df):
        for path_i, path_j in zip(paths, paths[1:]):
            path_i_real = os.path.join('data/logos/', path_i)
            path_j_real = os.path.join('data/logos/', path_j)

            # Test i->j
            similarities = compute_similarity(path_i_real, embeddings, embeddings_paths).cpu().numpy().flatten()
            top_images = np.argsort(similarities)[::-1]
            top_n = top
            top_paths = [embeddings_paths[i] for i in top_images[:top_n]]
            total_checks += 1
            if os.path.join('data/logos/', path_j) not in top_paths:
                misses.append((path_i_real, path_j_real))

            # Test j->i
            similarities = compute_similarity(path_j_real, embeddings, embeddings_paths).cpu().numpy().flatten()
            top_images = np.argsort(similarities)[::-1]
            top_n = top
            top_paths = [embeddings_paths[i] for i in top_images[:top_n]]
            total_checks += 1
            if os.path.join('data/logos/', path_i) not in top_paths:
                misses.append((path_j_real, path_i_real))

    print(f'Samples: {total_checks}, Misses: {len(misses)}')
    accuracy = 1.0 - len(misses) / total_checks
    print(f'^- Accuracy: {accuracy * 100.0:.2f}%')


def evaluate_model(embeddings, embeddings_paths, top):
    print('Evaluating the model...')

    print('1. Testing (validation part)...')
    evaluate_model_on('data/similar_valid.csv', embeddings, embeddings_paths, top)

    print('2. Testing (full dataset)...')
    evaluate_model_on('data/similar.csv', embeddings, embeddings_paths, top)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings-path', type=str)
    parser.add_argument('--top', type=int)
    args = parser.parse_args()

    print('Running on device', device)

    top = args.top
    if top is None:
        top = 10

    embeddings_path = args.embeddings_path
    if embeddings_path is None:
        embeddings_path = 'models/logos_embedding.pt'
    embeddings_path_root, _ = os.path.splitext(embeddings_path)
    embeddings_paths_path = embeddings_path_root + '.csv'
    if not os.path.exists(embeddings_path):
        print(f'Embeddings at `{embeddings_path}` not found')
        sys.exit(1)
    if not os.path.exists(embeddings_paths_path):
        print(f'Embeddings paths csv file at `{embeddings_paths_path}` not found')
        sys.exit(1)

    embeddings = torch.load(embeddings_path)
    embeddings_paths = pd.read_csv(embeddings_paths_path, header=None, index_col=None)[0].tolist()

    evaluate_model(embeddings, embeddings_paths, top)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
