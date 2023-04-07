# Guitar Style Dataset

In this work we present an original, publicly available audiovisible [dataset](https://github.com/magcil/guitar_style_dataset/tree/main/data) for guitar playing style classification, which is associated with the distinction across 9 types of playing techniques that encompass the vast majority of guitar technique types. We provide two different ways for classifying the data that can help you comprehend how these categories are separated.

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

The basic script is the `train.py`. You can download the folders containing the wav files from [here](https://github.com/magcil/guitar_style_dataset/tree/1-svm/data/wav) and run the script as shown below:

```
python3 train.py -w '/guitar_style_dataset/data/alternate picking' '/guitar_style_dataset/data/legato' '/guitar_style_dataset/data/tapping' '/guitar_style_dataset/data/sweep picking' '/GitHub/guitar_style_dataset/data/vibrato' '/guitar_style_dataset/data/hammer on' '/guitar_style_dataset/data/pull off' '/guitar_style_dataset/data/slide' '/guitar_style_dataset/data/bend'
```

You can use the flag `-rf` to work with predefined folds described in `folds.json`.

```
python3 train.py -w '/guitar_style_dataset/data/alternate picking' '/guitar_style_dataset/data/legato' '/guitar_style_dataset/data/tapping' '/guitar_style_dataset/data/sweep picking' '/GitHub/guitar_style_dataset/data/vibrato' '/guitar_style_dataset/data/hammer on' '/guitar_style_dataset/data/pull off' '/guitar_style_dataset/data/slide' '/guitar_style_dataset/data/bend' -rf folds.json
```
