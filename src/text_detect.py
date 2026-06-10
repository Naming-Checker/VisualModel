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


def load_detect_model(model_path=None):
    if model_path is None:
        model_path = 'models/text_detect.safetensors'
        print(f"Loading model from default path: \'{model_path}\'")

    model = models.vgg16()
    model.classifier.append(nn.Linear(1000, 2)) # 2 classes: text, logo
    model.classifier.append(nn.Sigmoid())

    load_model(model, model_path)

    model = model.to(device)
    model.eval()
    return model


# Use this for API calls
#
# Returns (float, float)
# tuple of confidence values (in range 0..=1) of pure text vs visual logo.
def calculate_text_likelihood(image_path, model):
    image = load_image(image_path)
    classification = model(image.unsqueeze(0).to(device)).detach().cpu().numpy().flatten()
    return (classification[0], classification[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str)
    parser.add_argument('--model', type=str)
    # parser.add_argument('--embeddings-path', type=str)
    args = parser.parse_args()

    print('Running on device', device)

    model = load_detect_model(args.model)
    text, logo = calculate_text_likelihood(args.image_path, model)
    print(f"Pure Text: {text*100.0:.2f}%")
    print(f"Visual Logo: {logo*100.0:.2f}%")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
