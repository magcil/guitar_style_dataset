import os
import sys
sys.path.append('..')
import numpy as np
import pandas as pd

from utils.feature_extraction import feature_extractor
from utils.utils import kfold_cross_val, leave_one_metadata_out, ready_folds_train, plot_cm
import re

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


def data_preparation_and_train(wav_path, fold, ready_folds=None):
    
    num_of_classes = len(wav_path)
    
    # STEP 1: feature extraction process
    if os.path.exists(f'features_{num_of_classes}_classes.npy'):
        # check if feature extraction has already been performed
        print(f"\nFeature extraction for {wav_path} has already been performed. Continuing...\n")
        
        features_list = np.load(f'features_{num_of_classes}_classes.npy')
        file_names = np.load(f'wav_names_{num_of_classes}_classes.npy')
        shapes_list = np.load(f'shapes_{num_of_classes}_classes.npy')
    else:
        features_list, class_names, file_names, shapes_list = feature_extractor(wav_path, num_of_classes)
        
    if len(features_list) > 0:
        label_mapping = [class_mapping_dict[path.split('/')[-1]] for path in wav_path]
        
        labels = []
        # create list of labels
        for count, name in zip(shapes_list, label_mapping):
            labels.extend([name]*count)
    else:
        raise ValueError("Features' list does not contain elements.")
    
    # STEP 2: cross-validation (k-fold or leave-one-out method)
    if ready_folds is not None:
        cm = ready_folds_train(file_names, labels, features_list, ready_folds)  
        print(class_mapping_dict)
        class_names = list(class_mapping_dict.keys())
        plot_cm(cm, class_names)
        
        
    else:
        if fold.isdigit():
            # features_list: list of feature vectors
            # labels: list of labels
            fold = int(fold)
            kfold_cross_val(features_list, file_names, labels, fold)
        
        elif fold=="guitar" or fold=="amplifier" or fold=="amp":
            # 1st col: wav_names, 2nd col: labels, the rest cols represent the features
            file_names = [os.path.basename(wav_name) for wav_name in file_names]
            df = pd.DataFrame({
                'file_name': file_names,
                'label': labels
            })
            
            features_list = pd.DataFrame(features_list.tolist())
            df = pd.concat([df, features_list], axis=1)
            
            # add 2 columns for guitar and amplifier names
            df['guitar'] = df['file_name'].str.split('_').str[2]
            df['amplifier'] = df['file_name'].str.split('_').str[3]
            
            
            if fold=="guitar" or fold=="amplifier" or fold=="amp":
                if fold=="amp":
                    fold = "amplifier"
                    
                leave_one_metadata_out(df, fold)
                print(class_mapping_dict)
                
            # print(df)

        else:
            raise ValueError("fold must either be a number or a string (guitar or amplifier) to choose between kfold or leave-one-out cross-validation.")