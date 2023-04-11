import os
import sys
sys.path.append('..')
import numpy as np
import warnings
import argparse
import pandas as pd

from data_load.dataloading import data_preparation_and_train

"""
Examples:

python3 train.py -d data/wav -j data/folds.json
python3 train.py -d data/wav -j data/guitars.json
python3 train.py -d data/wav -j data/amplifiers.json

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
        help="The path to the directories containing the WAV files.")
    
    parser.add_argument(
        "-f",
        "--fold",
        type=str,
        default="5",
        help="Random kfold (deafult k=5) cross validation.",
    )
    
    parser.add_argument(
        "-j",
        "--json_folds",
        type=str,
        required=False,
        help="The json file containing the train-test folds."
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    
    args = parse_arguments()
    data_path = args.data_path
    fold = args.fold
    json_folds = args.json_folds
    
    class_folders = []
    
    for folder in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, folder)):
            class_folders.append(os.path.join(data_path, folder))
    
    print("Guitar Style Classes: ", [os.path.basename(folder) for folder in class_folders])
    
    if 'guitar' in json_folds:
        print("Leaving one guitar out in corss-validation. . .")
    elif 'amplifier' in json_folds:
        print("Leaving one amplifier out in cron-validation. . .")
        
    # SVM training
    data_preparation_and_train(class_folders, fold, json_folds)
    
    