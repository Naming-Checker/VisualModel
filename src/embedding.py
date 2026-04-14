import os
import sys

import argparse
import csv
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


def load_similarity_model():
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(device)
    model.eval()
    return model


def generate_embeddings(model, loader):
    model = model.to(device)
    embeddings = torch.tensor([]).to(device)
    all_paths = []
    print("Generating embeddings...")
    for images, paths in tqdm(loader):
        encoding = model(images.to(device)).detach()
        embeddings = torch.cat((embeddings, encoding))
        all_paths += paths
    return embeddings, all_paths


def load_paths(root_dir) -> list:
    filenames = next(os.walk(root_dir), (None, None, []))[2]
    return [os.path.join(root_dir, filename) for filename in filenames]


class LogosDataset(Dataset):
    def __init__(self, root_dir, frames_limit=None, transform=None):
        self.root = root_dir
        self.transform = transform

        self.images = []

        images = load_paths(root_dir)
        image_paths = []

        if frames_limit is None:
            image_paths = images
        else:
            np.random.shuffle(images)
            image_paths.extend(images[:frames_limit])

        self.images.extend(image_paths)

        print(f'loaded {len(self.images)} paths')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image, img_path


def data_loader(data_dir, batch_size, frames_limit=None, shuffle=True):
    transform = image_transform()
    dataset = LogosDataset(data_dir, frames_limit, transform)

    indices = list(range(len(dataset)))
    if shuffle:
        np.random.shuffle(indices)

    sampler = SubsetRandomSampler(indices)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--logos-path', type=str)
    parser.add_argument('--logos-limit', type=int)
    args = parser.parse_args()

    print('Running on device', device)

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 16


    logos_path = args.logos_path
    if logos_path is None:
        logos_path = 'data/logos'
    if not os.path.exists(logos_path):
        print(f'Logos at `{logos_path}` not found')
        sys.exit(1)

    loader = data_loader(logos_path, batch_size, frames_limit=args.logos_limit)

    model = load_similarity_model()

    embeddings, embedding_paths = generate_embeddings(model, loader)

    print('Saving embeddings to `models/logos_embedding.pt` and `models/logos_embedding.csv`')
    torch.save(embeddings, 'models/logos_embedding.pt')
    pd.DataFrame(embedding_paths).to_csv('models/logos_embedding.csv', header=None, index=False)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
