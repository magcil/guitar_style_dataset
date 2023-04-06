import os
import sys
sys.path.append('..')
import numpy as np
import pandas as pd

from utils.feature_extraction import feature_extractor
from utils.utils import kfold_cross_val
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


def data_preparation(wav_path, fold):
    
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
        
        unique_guitars = df['guitar'].unique().tolist()
        
        print(f"Guitars: {unique_guitars}")
        
        for guitar in unique_guitars:
            train_guitars = unique_guitars.copy()
            train_guitars.remove(guitar)
            print(f"Guitars for training: {train_guitars}")
            print(f"Guitar for test: {[guitar]}")

            X_train = df[df['guitar'].isin(train_guitars)].iloc[:, 2:138]
            y_train = df[df['guitar'].isin(train_guitars)]['label']
            
            X_test = df[df['guitar'].isin([guitar])].iloc[:, 2:138]
            y_test = df[df['guitar'].isin([guitar])]['label']
            
            print(f"\n X_train: {len(X_train)} y_train: {len(y_train)}"+
                  f"\n X_test: {len(X_test)} y_test: {len(y_test)}\n")
        
        
        print(df)
        

   
    # elif fold=="guitar" or fold=="amplifier":
    #     metadata_dict = {}
    #     for folder_path in wav_path:            
    #         for file_name in os.listdir(folder_path):
    #             tmp = file_name.replace(".wav", "")
    #             metadata_list = tmp.split("_")
                
    #             # do not check 'exercise' in metadata list
    #             if len(metadata_list) == 6:
    #                 metadata_list.pop()
    #             metadata_list.pop()
    #             _, class_id, guitar, amp = metadata_list
    #             # print(f"list: {guitar} and song name: {file_name}")
                
    #             if fold=="guitar":
    #                 if guitar not in metadata_dict.keys():
    #                     metadata_dict[guitar] = []
    #                 metadata_dict[guitar].append(file_name)
    #             elif fold=="amplifier" or fold=="amp":
    #                 if amp not in metadata_dict.keys():
    #                     metadata_dict[amp] = []
    #                 metadata_dict[amp].append(file_name)

    else:
        raise ValueError("fold must either be a number or a string (guitar or amplifier) to choose between kfold or leave-one-out cross-validation.")