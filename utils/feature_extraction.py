import os
import numpy as np
import sys
sys.path.append('..')

from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import MidTermFeatures as aF
from pyAudioAnalysis import ShortTermFeatures as sF

def feature_extractor(dirs: str, 
                      num_of_classes: int,
                      mid_window: float = 1.0, 
                      mid_step: float = 1.0, 
                      short_window: float = 0.05,
                      short_step: float = 0.05,
                      ):
    
    """
    Feature extraction function using the pyAudioAnalysis library.

    Returns:
        features: list of features per folder
        class_names: list of class names based on folder names
        file_names: list of full path file names
    """

    features, class_names, file_names = \
        aF.multiple_directory_feature_extraction(dirs, mid_window, mid_step, short_window, short_step)

    file_names = [item for sublist in file_names for item in sublist]

    list_of_ids = None
    if len(features) == 0:
        print("trainSVM_feature ERROR: No data found in any input folder!")
        return

    n_feats = features[0].shape[1]
    feature_names = ["features" + str(d + 1) for d in range(n_feats)]

    for i, feat in enumerate(features):
        if len(feat) == 0:
            print("trainSVM_feature ERROR: " + dirs[i] +
                    " folder is empty or non-existing!")
            return
    
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
    
    return features_list, class_names, file_names, shapes_list
            