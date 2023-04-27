# Guitar Style Dataset

In this work we present an original, publicly available audiovisible dataset for guitar playing style classification, which is associated with the distinction across 9 types of playing techniques that encompass the vast majority of guitar technique types. We provide two different ways for classifying the data that can help you comprehend how these categories are separated.

## 1. Setup

It is preferable to work on a virtual environment

```
virtualenv env
source env/bin/activate
```
Once the virtual environment is activated, install the requirements:
```
pip3 install -r requirements.txt
```

## 2. Train
### 2.1 SVM 

The basic script is the `train.py`. To train an `SVM` model on the data using kFold cross-validation _(default k=5)_, run the following script:
```
python3 train.py -d data/wav/
```

> where `data/wav` is the directory which contains the class-folders with the wav files. 

Once you run the code, the feature vectors are saved in `.npy` files, so that on the next run, the feature extraction process is omitted.

You can change the value of `k` using the flag `-f`. Example:
```
python3 train.py -d data/wav/ -f 10
```

Some `.json` files have been created inside the `data` folder.
- `amplifiers.json` is a dictionary with 3-folds based on the 3 different amplifiers (leave-one-amplifier-out in each fold's test)
- `guitars.json` is a dictionary with 3-folds based on the 3 different guitars (leave-one-guitar-out in each fold's test)
- `folds.json` is a dictionary with 5-folds based on the exercises 

You can use the flag `-j` to work with the predefined folds described in the json files. Example:

```
python3 train.py -d data/wav -j data/folds.json
```

The results of each fold, as well as the aggregated ones, are both printed in the console and in `.txt` files.

### 2.2 CNNs
You can also train `CNNs` using Mel-spectrograms that correspond to segments of the wav files of the dataset.

#### 2.2.1 Predictions on wav files using majority vote

For training `CNNs` and testing them based on predictions on full wav files, you can run the script below:
```
python3 deep_audio_features_wrapper/deep_audio_features_test.py -i data/wav/ -j data/folds.json -s 1 -o /path/to/output_folder
```
 where: 
 - `data/wav/` is the path to the `wav` folder, that contains all wav files of the dataset
 - `data/folds.json` is the path to a json file containing the fold splits to be used for training the networks
 - `/path/to/output_folder` is the path to an empty folder where the segments of the wav files will be placed
 -  `-s` determines the segment size of wav files that will be used for training, in seconds

#### 2.2.2 Segment-level predictions

For segment-level predictions you can add the flag `-t`. In this case, the following command should be executed:

```
python3 deep_audio_features_wrapper/deep_audio_features_test.py -i data/wav/ -j data/folds.json -s 1 -o /path/to/output_folder -t
```

For both options, the results for each fold and the final results will be printed in the console as well as in `.txt`  files.
