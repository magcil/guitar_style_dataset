import os
import sys
sys.path.append('..')
import numpy as np
import warnings
import argparse
import pandas as pd

from data_load.dataloading import data_preparation_and_train

"""

python3 train.py -w '/media/antonia/Seagate/GitHub/guitar_style_dataset/data/alternate picking' '/media/antonia/Seagate/GitHub/guitar_style_dataset/data/legato' '/media/antonia/Seagate/GitHub/guitar_style_dataset/data/tapping' '/media/antonia/Seagate/GitHub/guitar_style_dataset/data/sweep picking' '/media/antonia/Seagate/GitHub/guitar_style_dataset/data/vibrato' '/media/antonia/Seagate/GitHub/guitar_style_dataset/data/hammer on' '/media/antonia/Seagate/GitHub/guitar_style_dataset/data/pull off' '/media/antonia/Seagate/GitHub/guitar_style_dataset/data/slide' '/media/antonia/Seagate/GitHub/guitar_style_dataset/data/bend' -rf folds.json 


"""

def parse_arguments():
    """
    Parse arguments for training.
    """
    parser = argparse.ArgumentParser(description="Guitar Style Classifcation")

    parser.add_argument(
        "-w", 
        "--wav_path", 
        required=True, 
        action='append',
        nargs='+', 
        help="The paths to the WAV files.")
    
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
    wav_path = args.wav_path
    fold = args.fold
    ready_folds = args.ready_folds
    
    wav_path = [item for sublist in wav_path for item in sublist]

    data_preparation_and_train(wav_path, fold, ready_folds)
    print("Guitar Style Classes: ", wav_path)
        
    