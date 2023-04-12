import os
import pickle
import wave
import contextlib
import subprocess

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


def get_label(filename: str) -> str:
    prev_str = 'class_'
    idx = filename.rfind(prev_str)
    idx += len(prev_str)
    label = filename[idx]
    return label


def get_wav_duration(fname):
    with contextlib.closing(wave.open(fname, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration


def prepare_dirs(input_dir, train_wavs, test_wavs, output_path, segment_size, test_seg):
    """Given a train/test split create dirs with segmented wavs on train separated in classes"""
    train_path = os.path.join(output_path, 'train')
    test_path = os.path.join(output_path, 'test')
    temp_wav_path = os.path.join(output_path, "temp.wav")
    input_files = os.listdir(input_dir)
    os.mkdir(train_path)
    os.mkdir(test_path)
    train_path_dict = {}
    for label, guitar_technique in CLASS_MAPPING.items():
        tmp_train_path = os.path.join(train_path, guitar_technique)
        os.mkdir(tmp_train_path)
        train_path_dict[label] = tmp_train_path
    for wav_file in input_files:
        wav_name = wav_file.rstrip(".wav")
        wav_path = os.path.join(input_dir, wav_file)
        if wav_file in train_wavs:
            label = get_label(wav_file)
            out_path = train_path_dict[int(label)]
            fnc = 'train'
        elif wav_file in test_wavs:
            out_path = test_path
            fnc = 'test'
        else:
            fnc = 'other'
            print(f'File {wav_file} does not belong to train nor test set. \nSkipping {wav_file}.')
        # get wav duration
        if 'out_path' in locals():
            try:
                dur = get_wav_duration(os.path.join(input_dir, wav_file))
            except Exception as err:
                raise err
        else:
            pass
        # create a temporary wav that has been trimmed accordingly depending on the segment size
        if 'dur' in locals() and (fnc == 'train' or fnc == 'test'):
            if fnc == 'train' or test_seg == True:
                end = (dur // segment_size) * segment_size
                try:
                    subprocess.check_call(
                        [
                            "ffmpeg", "-i", wav_path, "-ss", "0", "-to",
                            str(end), "-ar", "8000", "-ac", "1", "-y", "-loglevel", "quiet", temp_wav_path
                        ]
                    )
                except subprocess.CalledProcessError as e:
                    print(f"An error occured saving the temp wav of {wav_file}.\nError: {e}")
                    print(f"Skipping the splitting of {wav_file}.")
                    continue
                # segment temporary wav and save it in corresponding directory
                try:
                    subprocess.check_call(
                        [
                            "ffmpeg", "-i", temp_wav_path, "-f", "segment", "-segment_time",
                            str(segment_size), "-ar", "8000", "-ac", "1", "-loglevel", "quiet",
                            f"{out_path}/{wav_name}_{segment_size}_%03d.wav"
                        ]
                    )
                except subprocess.CalledProcessError as e:
                    print(f"An error occured with the segmentation of {wav_file}.\nError: {e}")
            elif fnc == 'test':
                end = (dur // segment_size) * segment_size
                test_wav_path = os.path.join(out_path, f"{out_path}/{wav_name}_trimmed.wav")
                try:
                    subprocess.check_call(
                        [
                            "ffmpeg", "-i", wav_path, "-ss", "0", "-to",
                            str(end), "-ar", "8000", "-ac", "1", "-y", "-loglevel", "quiet", test_wav_path
                        ]
                    )
                except subprocess.CalledProcessError as e:
                    print(f"An error occured saving the trimmed {wav_file} file.\nError: {e}")
                    print(f"Skipping {wav_file}.")
                    continue
            else:
                pass
        else:
            pass

def deep_audio_training(output_path):
    """Train on dirs using deep audio features"""

    train_path = os.path.join(output_path, 'train')
    train_dirs = next(os.walk(train_path))[1]
    train_dirs = [os.path.join(output_path, 'train', dir) for dir in train_dirs]
    bt.train_model(train_dirs, 'technique_classifier')


def validate_on_test(output_path, test_seg, model_path='pkl/technique_classifier.pt'):
    """Validate on test using deep audio features"""
    test_path = os.path.join(output_path, 'test')
    y_true, y_pred = [], []

    test_songs = crawl_directory(test_path)
    with open(model_path, 'rb') as f:
        model_params = pickle.load(f)
    model_class_mapping = model_params['classes_mapping']

    for song in test_songs:
        if os.path.basename(song) == 'temp_trimmed.wav':
            continue
        if test_seg == False:
            preds, posteriors = btest.test_model(model_path, song, layers_dropped=0, test_segmentation=True)
            probs = np.exp(posteriors) / np.sum(np.exp(posteriors), axis=1).reshape(posteriors.shape[0], 1)
            p_aggregated = probs.mean(axis=0)
            counts = np.bincount(preds)
            results = []
            for i in range(counts.size):
                results.append((counts[i], p_aggregated[i], i))
            results = sorted(results, key=lambda x: (x[0], x[1]), reverse=True)
            pred_label = results[0][-1]
            y_pred.append(model_class_mapping[pred_label])
            true_label = int(os.path.basename(song).split('_')[1])
            y_true.append(CLASS_MAPPING[true_label])
            print(
                f'True: {CLASS_MAPPING[true_label]} | Pred: {model_class_mapping[pred_label]}' +
                f'| Prob: {results[0][1]} | Counts: {results[0][0]}'
            )
        else:
            pred, posteriors = btest.test_model(model_path, song, layers_dropped=0, test_segmentation=False)
            y_pred.append(model_class_mapping[pred[0]])
            true_label = int(os.path.basename(song).split('_')[1])
            y_true.append(CLASS_MAPPING[true_label])
            print(
                f'True: {CLASS_MAPPING[true_label]} | Pred: {model_class_mapping[pred[0]]}'
            )
            
    return y_true, y_pred