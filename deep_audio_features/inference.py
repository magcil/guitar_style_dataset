import argparse
import pickle
import os
import glob

import numpy as np

from deep_audio_features.bin import basic_test as btest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input_wav', required=True, help='The guitar solo to be predicted in splitted in segments of 1 second.'
    )
    parser.add_argument('-m', '--model', required=True, help='The path of the pt file.')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    input_wav = args.input_wav
    model = args.model

    with open(model, 'rb') as f:
        model_params = pickle.load(f)
    model_class_mapping = model_params['classes_mapping']

    segmented_wavs = sorted(list(glob.glob(os.path.join(input_wav, '*.wav'))))
    results = []
    for segment in segmented_wavs:
        d, p = btest.test_model(model, segment, test_segmentation=False)
        p = np.squeeze(p)
        p = np.exp(p) / np.sum(np.exp(p))
        sorts = np.argsort(p)
        results.append((sorts[-1], sorts[-2]))
        
    
    print(f'\n\n {20*"*"} Results Per segment {20*"*"}\n\n')
    for i, pred_label in enumerate(results):
        pred_label_1, pred_label_2 = results[i]
        print(f'Sec {i}->{i+1}: {model_class_mapping[pred_label_1]}, {model_class_mapping[pred_label_2]}')