import os
import torch
import librosa
import json
from torch.utils.data import Dataset
from audio_utils import extract_mel_spectrogram, get_label

FILE_PATH = os.path.abspath(__file__)

class GuitarDataset(Dataset):
    '''
    Args: 
    data_path: path to a folder containing all wav files (of all classes)
    folds_json: json file that contains the train and test sets for each fold
    fold_num: number of current fold
    Returns:
    X: Mel Spectrogram
    y: class label
    '''
    def __init__(self, data_path:str, folds_json:str, fold_num:int, transform=None):
        self.data_path = data_path
        self.folds_json_path = folds_json
        self.fold_num = fold_num
        self.transform = transform
        self.segmented_data = {}
        self.get_indices()
    
    def get_indices(self):
        '''
        Creates a dictionary that contains:
        keys: the name of each segment
        values: the time index (sec) that is the start of each segment
        '''
        folds_json_path = self.folds_json_path
        fold_num = self.fold_num 
        if fold_num in range(5):
            pass
        else:
            raise ValueError('Fold number must lie between 0 and 4')
        fold = f'fold_{fold_num}'
        with open(folds_json_path) as fold_j:
            folds_json = json.load(fold_j)
        for wav in folds_json[fold]["train"]:
            wav_path = os.path.abspath(os.path.join(self.data_path, wav))
            try:
                signal, sr = librosa.load(wav_path, sr=8000, mono=True)
            except Exception as err:
                log_info = f'Error occured on {wav_path}.'
                print(log_info)
                print(f"Exception: {err}")
                print(f'Removed filename: {wav}')
            else:
                max_time_idx = int(signal.size / sr)
                if max_time_idx:
                    for idx in range(max_time_idx):
                        self.segmented_data[f'{wav}_seg_{idx}'] = idx
        else:
            pass

    def __len__(self):
        return len(self.segmented_data)
    
    def __getitem__(self, idx):
        segmented_data = self.segmented_data
        filenames = list(segmented_data.keys())
        current_seg = filenames[idx]
        time_idx = segmented_data[current_seg]
        wav_path = os.path.abspath(os.path.join(self.data_path, current_seg[:-6]))
        signal, sr = librosa.load(wav_path, sr=8000, mono=True)
        signal_seg = signal[time_idx * sr: (time_idx + 1) * sr]
        X = extract_mel_spectrogram(signal_seg)
        y = get_label(current_seg)
        return X, y