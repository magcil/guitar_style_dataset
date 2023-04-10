import argparse
import json
import os
import shutil

from sklearn.metrics import accuracy_score, recall_score, precision_score

from deep_audio_utils import prepare_dirs, deep_audio_training, validate_on_test, crawl_directory


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_data', required=True, help='Path to the recorded guitar audio wav files.')
    parser.add_argument('-j', '--json_file', required=True, help='The json file containing the 5-fold splits.')
    parser.add_argument(
        '-s', '--segment_size', type=int, default=1, help='The segment size of wav files in training (secs).'
    )
    parser.add_argument('-o', '--output_dir_path', required=True, help='Path to store the train/test splits.')

    return parser.parse_args()


if __name__ == '__main__':
    """Train and validate on folds using deep audio features."""

    args = parse_args()
    input_data_path = args.input_data
    json_file = args.json_file
    segment_size = args.segment_size
    output_path = args.output_dir_path

    with open(json_file, 'r') as f:
        splits = json.load(f)
    songs = crawl_directory(input_data_path)
    accs, recalls, precisions, f1_scores = [], [], [], []
    for fold in splits:
        train_wavs, test_wavs = [], []
        # Split to train/test
        train_set = splits[fold]['train']
        test_set = splits[fold]['test']
        for song in songs:
            if os.path.basename(song) in train_set:
                train_wavs.append(song)
            else:
                test_wavs.append(song)
        print('Fold:', fold)
        print('Number in test:', len(test_wavs))
        print('Number in train:', len(train_wavs))

        print(f'Preparing dirs...')
        prepare_dirs(train_wavs, test_wavs, output_path, segment_size)
        print('Training starts...')
        class_mapping = deep_audio_training(output_path)
        y_true, y_pred = validate_on_test(output_path, class_mapping)

        acc = 100 * accuracy_score(y_true, y_pred)
        precision = 100 * precision_score(y_true, y_pred)
        recall = 100 * recall_score(y_true, y_pred)
        f1_score = 100 * (precision * recall) / (precision + recall)

        print(
            f'Results on {fold}\nAccuracy: {acc:.2f}\nRecall: {recall:.2f}'
            f'\nPrecision: {precision:.2f}\nF1: {f1_score:2f}'
        )
        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

        # Remove dirs
        shutil.rmtree(os.path.join(output_path, "train"))
        shutil.rmtree(os.path.join(output_path, "test"))
