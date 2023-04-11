import os
import sys
sys.path.append('..')
import numpy as np
import pandas as pd

from utils.feature_extraction import feature_extractor
from utils.utils import kfold_cross_val, custom_folds_train, plot_cm


class_mapping_dict = {
    'alternate picking': 0,
    'legato': 1,
    'tapping': 2,
    'sweep picking': 3,
    'vibrato': 4,
    'hammer on': 5,
    'pull off': 6,
    'slide': 7,
    'bend': 8 
}


def data_preparation_and_train(wav_path, fold, json_folds=None):
    
    num_of_classes = len(wav_path)
    
    # STEP 1: feature extraction process
    if os.path.exists(f'features_{num_of_classes}_classes.npy'):
        # check if feature extraction has already been performed
        wav_classes = [os.path.basename(folder) for folder in wav_path]
        print(f"\nFeature extraction for {wav_classes} has already been performed. Continuing...\n")
        
        features_list = np.load(f'features_{num_of_classes}_classes.npy')
        file_names = np.load(f'wav_names_{num_of_classes}_classes.npy')
        shapes_list = np.load(f'shapes_{num_of_classes}_classes.npy')
    else:
        features_list, class_names, file_names, shapes_list = feature_extractor(wav_path, num_of_classes)
        features_list = np.array(features_list)
    
    if (features_list.shape[0]) > 0:
        label_mapping = [class_mapping_dict[path.split('/')[-1]] for path in wav_path]
        labels = []
        # create list of labels (labels as many as the shapes (from shapes_list))
        for count, label in zip(shapes_list, label_mapping):
            labels.extend([label]*count)
    else:
        raise ValueError("Features' list does not contain elements.")
    
    # STEP 2: cross-validation (k-fold or leave-one-out method)
    if json_folds is not None:
        # leave-one-guitar/amplifier/exercise out
        cm = custom_folds_train(file_names, labels, features_list, json_folds)  
        class_names = list(class_mapping_dict.keys())
        plot_cm(cm, class_names, json_folds)
    
    else:
        if fold.isdigit():
            # random kfold
            fold = int(fold)
            cm = kfold_cross_val(file_names, labels, features_list, fold)
            class_names = list(class_mapping_dict.keys())
            plot_cm(cm, class_names)

        else:
            raise ValueError("fold must either be a number or a string (guitar or amplifier) to choose between kfold or leave-one-out cross-validation.")