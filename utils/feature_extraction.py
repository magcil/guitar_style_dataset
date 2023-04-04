import os
import numpy as np
import sys
sys.path.append('..')

from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import MidTermFeatures as aF
from pyAudioAnalysis import ShortTermFeatures as sF

def feature_extractor(dirs: str, 
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
    
    return features, class_names, file_names
            