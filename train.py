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
        '-s', 
        '--segment_size', 
        type=int, 
        default=1, 
        help='The segment size of wav files in training in seconds (default: 1)'
    )

    parser.add_argument(
        '-t', 
        '--test_seg', 
        type=bool, 
        default=False, 
        help='For segment-level predictions',
        action=argparse.BooleanOptionalAction
    )

    parser.add_argument(
        '-o', 
        '--output_dir', 
        required=True, 
        help='Path to store the train/test splits.'
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
    segment_size = args.segment_size
    output_path = args.output_dir
    test_seg = args.test_seg
    
    try:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(f"Directory '{output_path}' created successfully.")
        else:
            print(f"Directory '{output_path}' already exists.")
    except Exception as e:
        print(f"Error creating directory: {e}")
    
    if json_folds is not None:
        if 'guitar' in json_folds:
            print("Leaving one guitar out in corss-validation. . .")
        elif 'amplifier' in json_folds:
            print("Leaving one amplifier out in cron-validation. . .")

    # SVM training
    data_preparation_and_train(data_path, fold, json_folds, segment_size, test_seg, output_path)
    
    
