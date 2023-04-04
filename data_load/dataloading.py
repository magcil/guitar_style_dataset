import os
import sys
sys.path.append('..')
import numpy as np

from utils.feature_extraction import feature_extractor
from utils.utils import kfold_cross_val


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
    
    # feature extraction process
    if os.path.exists(f'features_{num_of_classes}_classes.npy'):
        # check if feature extraction has already been performed
        print(f"\nFeature extraction for {wav_path} has already been performed. Continuing...")
        
        features_list = np.load(f'features_{num_of_classes}_classes.npy')
        file_names = np.load(f'wav_names_{num_of_classes}_classes.npy')
        shapes_list = np.load(f'shapes_{num_of_classes}_classes.npy')
    else:
        features, class_names, file_names = feature_extractor(wav_path)
        shapes_list = [arr.shape[0] for arr in features]

        feat = np.concatenate(features, axis=0)
        features_list = feat.tolist()

        print(f"\n There are: {len(features_list)} feature vectors.")
        
        # features_list contains all the feature vectors
        np.save(f'features_{num_of_classes}_classes.npy', features_list)
        # file_names contains all the corresponding file names
        np.save(f'wav_names_{num_of_classes}_classes.npy', file_names)
        # shape_list contains the number of files in each class
        np.save(f'shapes_{num_of_classes}_classes.npy', shapes_list)
    
    if len(features_list) > 0:
        
        label_mapping = [class_mapping_dict[path.split('/')[-1]] for path in wav_path]
        
        labels = []
        # create list of labels
        for count, name in zip(shapes_list, label_mapping):
            labels.extend([name]*count)
    else:
        raise ValueError("Features list does not contain elements.")
    
    # cross-validation (k-fold or leave-one-out method)
    if fold=="guitar" or fold=="amplifier":
        metadata_dict = {}
        for folder_path in wav_path:            
            for file_name in os.listdir(folder_path):
                tmp = file_name.replace(".wav", "")
                metadata_list = tmp.split("_")
                
                # do not check 'exercise' in metadata list
                if len(metadata_list) == 6:
                    metadata_list.pop()
                metadata_list.pop()
                
                _, class_id, guitar, amp = metadata_list
                # print(f"list: {guitar} and song name: {file_name}")
                
                if fold=="guitar":
                    if guitar not in metadata_dict.keys():
                        metadata_dict[guitar] = []
                    metadata_dict[guitar].append(file_name)
                elif fold=="amplifier":
                    if amp not in metadata_dict.keys():
                        metadata_dict[amp] = []
                    metadata_dict[amp].append(file_name)
                
        print(len(metadata_dict))
        
        
    elif fold.isdigit():
        # features_list: list of feature vectors
        # labels: list of labels
        fold = int(fold)
        kfold_cross_val(features_list, file_names, labels, fold)
        
        
    else:
        raise ValueError("fold must either be a number or a string (guitar or amplifier) to choose between kfold or leave-one-out cross-validation.")