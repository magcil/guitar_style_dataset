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

The basic script is the `train.py`. To train an `SVM` model on the data, run the following script:
```
python3 train.py -d data/wav/
```

> where `data/wav` is the directory which contains the class-folders with the wav files.

You can use the flag `-rf` to work with predefined folds described in `data/folds.json`.

```
python3 train.py -d data/wav -rf data/folds.json
```
