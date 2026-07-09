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


# Convert analysis result to a tuple of values/weights
def to_distribution(info: ImageInfo):
    values = []
    weights = []
    for color in info.colors:
        (r,g,b) = color.rgb
        rgb = [r / 255.0, g / 255.0, b / 255.0]
        values.append(rgb)
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
