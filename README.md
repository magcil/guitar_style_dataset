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
