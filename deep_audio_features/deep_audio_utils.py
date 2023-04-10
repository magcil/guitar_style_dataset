import os
import pickle

import torch
from torch.utils.data import DataLoader
import numpy as np

from deep_audio_features.bin import basic_training as bt
from deep_audio_features.bin import basic_test as btest

CLASS_MAPPING = {
    0: 'alternate picking',
    1: 'legato',
    2: 'tapping',
    3: 'sweep picking',
    4: 'vibrato',
    5: 'hammer on',
    6: 'pull off',
    7: 'slide',
    8: 'bend'
}


def crawl_directory(directory: str, extension: str = None) -> list:
    """Crawling data directory
    Args:
        directory (str) : The directory to crawl
    Returns:
        tree (list)     : A list with all the filepaths
    """
    tree = []
    subdirs = [folder[0] for folder in os.walk(directory)]

    for subdir in subdirs:
        files = next(os.walk(subdir))[2]
        for _file in files:
            if extension is not None:
                if _file.endswith(extension):
                    tree.append(os.path.join(subdir, _file))
            else:
                tree.append(os.path.join(subdir, _file))
    return tree


def prepare_dirs(train_wavs, test_wavs, output_path, segment_size):
    """Given a train/test split create dirs with segmented wavs on train separated in classes"""
    train_path = os.path.join(output_path, 'train')
    test_path = os.path.join(output_path, 'test')

    # Prepare test dirs
    for label, guitar_technique in CLASS_MAPPING.items():

        pass
    pass


def deep_audio_training(output_path):
    """Train on dirs using deep audio features"""

    train_path = os.path.join(output_path, 'train')
    train_dirs = next(os.walk(train_path))[1]
    bt.train_model(train_dirs, 'technique_classifier.pt')

    return {i: technique for (i, technique) in enumerate(train_dirs)}


def validate_on_test(output_path, class_mapping, model_path='pkl/technique_classifier.pt'):
    """Validate on test using deep audio features"""
    test_path = os.path.join(output_path, 'test')
    y_true, y_pred = [], []
    test_songs = crawl_directory(test_path)

    for song in test_songs:
        preds, posteriors = btest.test_model(model_path, song, layers_dropped=0, test_segmentation=True)
        probs = np.exp(posteriors) / np.sum(np.exp(posteriors), axis=1).reshape(posteriors.shape[0], 1)
        p_aggregated = probs.mean(axis=0)
        preds = np.bincount(preds)
        res = []
        for i in range(preds.size):
            res.append((preds[i], p_aggregated[i], class_mapping[i]))
        res = sorted(res, key=lambda x: (x[0], x[1]), reverse=True)
        y_pred.append(res[0][2])
        song_label = int(os.path.basename(song).split('_')[1])
        y_true.append(class_mapping[song_label])

    return y_true, y_pred