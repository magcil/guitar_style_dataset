import argparse
import json
import os
import shutil

from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report

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
    logs = []

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
                train_wavs.append(os.path.basename(song))
            elif os.path.basename(song) in test_set:
                test_wavs.append(os.path.basename(song))
        print('Fold:', fold)
        print('Number in test:', len(test_wavs))
        print('Number in train:', len(train_wavs))
        logs.append(f'{5*"-"}> {fold} <{5*"-"}\n')

        print(f'Preparing dirs...')
        prepare_dirs(input_data_path, train_wavs, test_wavs, output_path, segment_size)
        print('Training starts...')
        deep_audio_training(output_path)
        y_true, y_pred = validate_on_test(output_path)
        labels = list(set(y_true))
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0, labels=labels)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0, labels=labels)
        f1_score = 2 * (precision * recall) / (precision + recall)

        print(
            f'Results on {fold}\nAccuracy: {100* acc:.2f}\nRecall: {100* recall:.2f}'
            f'\nPrecision: {100* precision:.2f}\nF1: {100 * f1_score:2f}'
        )
        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
        class_rep = classification_report(y_true, y_pred, labels=labels)
        logs.append(class_rep)

        # Remove dirs
        shutil.rmtree(os.path.join(output_path, "train"))
        shutil.rmtree(os.path.join(output_path, "test"))

    logs.append('Total Scores\n')
    final_results = f'Mean Accuracy: {100 * sum(accs) / len(accs)}\n' + \
    f'Mean Precision: {100 * sum(precisions) / len(precisions)}\n' + \
    f'Mean Recall: {100 * sum(recalls) / len(recalls)}\n' + \
    f'Mean F1 score: {100 * sum(f1_scores) / len(f1_scores)}\n'

    logs.append(final_results)
    print(final_results)

    with open('deep_audio_results.txt', 'w') as f:
        f.writelines(logs)
