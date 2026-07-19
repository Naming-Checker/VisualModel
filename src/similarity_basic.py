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


class SquarePad:
    def __call__(self, image):
        c, h, w = image.size()
        max_wh = np.max([w, h])
        hp = (max_wh - w) // 2 + max_wh // 10
        vp = (max_wh - h) // 2 + max_wh // 10
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 1, 'constant')


def image_transform():
    return transforms.Compose([
        transforms.ConvertImageDtype(torch.float),
        transforms.RGB(),
        SquarePad(),
        transforms.Resize((224,224)),
        transforms.Lambda(lambda x: torch.clamp(x, 0, 1))
    ])


def load_image(path):
    transform = image_transform()
    image = read_image(path)
    image = transform(image)
    return image


def compute_similarity(image_path, embeddings, model):
    image = load_image(image_path)
    encoded = model.to(device)(image.unsqueeze(0).to(device)).detach()

    similarity_fn = nn.CosineSimilarity()
    def similarity(other_enc):
        return similarity_fn(encoded, other_enc)
    similarity_vec = torch.vmap(similarity)
    return similarity_vec(embeddings)


def load_similarity_model(model_path=None):
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(device)
    model.eval()
    if model_path is not None:
        load_model(model, model_path)
    return model


# Use this for API calls
#
# Returns ([float], [str])
# tuple of similarity values and image paths (not sorted)
def calculate_similarity(image_path, model_path=None, embeddings_path=None):
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

    model = load_similarity_model(model_path)

    # Compute
    similarities = compute_similarity(image_path, embeddings, model).cpu().numpy().flatten()
    return similarities, embeddings_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--embeddings-path', type=str)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--top', type=int)
    args = parser.parse_args()

    print('Running on device', device)

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 16

    similarities, embeddings_paths = calculate_similarity(args.image_path, args.model, args.embeddings_path)
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
