import os
import sys
sys.path.append('..')
import numpy as np
import warnings
import argparse
import pandas as pd

from data_load.dataloading import data_preparation_and_train

"""

python3 train.py -d data/wav -rf data/folds.json

"""

def parse_arguments():
    """
    Parse arguments for training.
    """
    parser = argparse.ArgumentParser(description="Guitar Style Classifcation")

    parser.add_argument(
        "-d", 
        "--data_path", 
        required=True, 
        type=str,
        help="The paths to the directories containing the WAV files.")
    
    parser.add_argument(
        "-f",
        "--fold",
        type=str,
        default="5",
        help="Choose between K-Fold (default: k=5) and Leave-One-Out (guitar or amplifier) cross-validation",
    )
    
    parser.add_argument(
        "-rf",
        "--ready_folds",
        type=str,
        required=False,
        help="The directory with the predifined folds."
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    
    args = parse_arguments()
    data_path = args.data_path
    fold = args.fold
    ready_folds = args.ready_folds

    print("Guitar Style Classes: ", data_path)
    
    class_folders = []
    
    for folder in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, folder)):
            class_folders.append(os.path.join(data_path, folder))
    
    print("Classes: ", class_folders)
    data_preparation_and_train(class_folders, fold, ready_folds)
    
    