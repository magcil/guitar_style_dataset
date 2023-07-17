import os
import sys
sys.path.append('..')
import numpy as np
import pandas as pd

from utils.feature_extraction import feature_extractor
from utils.utils import kfold_cross_val, custom_folds_train, plot_cm, class_mapping_dict
from deep_audio_features_wrapper.deep_audio_utils import crawl_directory, prepare_dirs


def data_preparation_and_train(data_path, fold, json_folds, segment_size, test_seg, output_path):
    songs = crawl_directory(data_path, extension=".wav")

    class_folders = []
    
    for folder in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, folder)):
            class_folders.append(os.path.join(data_path, folder))

    print("Guitar Style Classes: ", [os.path.basename(folder) for folder in class_folders])

    wav_path = class_folders
    
    # cross-validation (k-fold or leave-one-out method)
    if json_folds is not None:
        # leave-one-guitar/amplifier/exercise out
        cm = custom_folds_train(json_folds, data_path, segment_size, test_seg, output_path)  
        class_names = list(class_mapping_dict.keys())
        plot_cm(cm, class_names, json_folds)
    
    # else:
    #     if fold.isdigit():
    #         # random kfold
    #         fold = int(fold)
    #         cm = kfold_cross_val(file_names, labels, features_list, fold)
    #         class_names = list(class_mapping_dict.keys())
    #         plot_cm(cm, class_names)

    #     else:
    #         raise ValueError("fold must either be a number or a string (guitar or amplifier) to choose between kfold or leave-one-out cross-validation.")
