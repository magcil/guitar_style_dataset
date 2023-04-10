import argparse

from deep_audio_utils import prepare_dirs, deep_audio_training, validate_on_test

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
    pass
