import os
import sys
sys.path.append('..')
import numpy as np
import warnings
import argparse
import pandas as pd

from data_load.dataloading import data_preparation

"""_summary_

python3 train.py -w '/media/antonia/Seagate/GitHub/guitar_style_dataset/toy/alternate picking' '/media/antonia/Seagate/GitHub/guitar_style_dataset/toy/legato' '/media/antonia/Seagate/GitHub/guitar_style_dataset/toy/sweep picking' '/media/antonia/Seagate/GitHub/guitar_style_dataset/toy/vibrato'

Returns:
    _type_: _description_
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

    return parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    
    args = parse_arguments()
    wav_path = args.wav_path
    fold = args.fold
    
    wav_path = [item for sublist in wav_path for item in sublist]
    
    # print("Guitar Style Classes: ", wav_path)
    
    data_preparation(wav_path, fold)
    
    
    
    
    