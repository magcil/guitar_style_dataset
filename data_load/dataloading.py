import os
import sys
sys.path.append('..')
import numpy as np
import pandas as pd

from utils.feature_extraction import feature_extractor
from utils.utils import kfold_cross_val, custom_folds_train_on_segments, plot_cm, class_mapping_dict#, custom_folds_train
from deep_audio_features_wrapper.deep_audio_utils import crawl_directory, prepare_dirs


def data_preparation_and_train(data_path, fold, json_folds, segment_size, test_seg, output_path):

    class_folders = []
    
    for folder in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, folder)):
            class_folders.append(os.path.join(data_path, folder))

    print("Guitar Style Classes: ", [os.path.basename(folder) for folder in class_folders])
    
    if segment_size >= 1:
        if json_folds is not None:
            # leave-one-guitar/amplifier/exercise out
            cm = custom_folds_train_on_segments(json_folds, data_path, segment_size, test_seg, output_path)  
            class_names = list(class_mapping_dict.keys())
            plot_cm(cm, class_names, json_folds)
    else:
        return "Define a segment size greater or equal to 1 sec."
    
