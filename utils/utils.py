import numpy as np
import pandas as pd
import os
import sys
import json
import matplotlib.pyplot as plt
import contextlib
import shutil
import re

from utils.feature_extraction import feature_extractor
from deep_audio_features_wrapper.deep_audio_utils import crawl_directory, get_wav_duration, get_label, prepare_dirs

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, make_scorer, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC, LinearSVC


class_mapping_dict = {
    'alternate picking': 0,
    'legato': 1,
    'tapping': 2,
    'sweep picking': 3,
    'vibrato': 4,
    'hammer on': 5,
    'pull off': 6,
    'slide': 7,
    'bend': 8 
}


def plot_cm(conf_matrix, class_names, folds=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    im = ax.imshow(conf_matrix, cmap='Blues')

    # Add annotations to the image
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, conf_matrix[i, j], ha='center', va='center', fontsize=8)

    # Add a colorbar legend to the plot
    fig.colorbar(im)
    ax.set_xticklabels([''] + class_names, fontsize=8, rotation=45, ha='right')
    ax.set_yticklabels([''] + class_names, fontsize=8)
    
    plt.xlabel('Predicted', fontsize=10)
    plt.ylabel('True', fontsize=10)

    if folds is not None:
        if 'guitar' in folds.lower():
            plt.title('Aggregated Confusion Matrix (Guitar)', fontsize=12)
            plt.savefig(f'guitars_{len(class_names)}_class_confusion_matrix.eps', format='eps')
            plt.savefig(f'guitars_{len(class_names)}_class_confusion_matrix.pdf', format='pdf')
        elif 'amplifier' in folds.lower():
            plt.title('Aggregated Confusion Matrix (Amplifier)', fontsize=12)
            plt.savefig(f'amplifiers_{len(class_names)}_class_confusion_matrix.eps', format='eps')
            plt.savefig(f'amplifiers_{len(class_names)}_class_confusion_matrix.pdf', format='pdf')
        else:
            plt.title('Aggregated Confusion Matrix', fontsize=12)
            plt.savefig(f'5_custom_folds_{len(class_names)}_class_confusion_matrix.eps', format='eps')
            plt.savefig(f'5_custom_folds_{len(class_names)}_class_confusion_matrix.pdf', format='pdf')
    else:
        plt.title('Aggregated Confusion Matrix', fontsize=12)
        plt.savefig(f'{len(class_names)}_class_confusion_matrix.eps', format='eps')
        plt.savefig(f'{len(class_names)}_class_confusion_matrix.pdf', format='pdf')

def create_df(file_names, labels, features_list):
    # ------- DataFrame structure -------
    # first column: wav names
    # second column: wav labels
    # the rest columns: feature vectors of each wav file
    
    file_names = [os.path.basename(wav_name) for wav_name in file_names]
    
    df = pd.DataFrame({
        'file_name': file_names,
        'label': labels
    })
    features_list = pd.DataFrame(features_list.tolist())
    df = pd.concat([df, features_list], axis=1)
    
    return df


def get_subfolders(path):
    subfolders = []

    for root, dirs, _ in os.walk(path):
        for dir_name in dirs:
            subfolders.append(os.path.join(root, dir_name))

    return subfolders


def majority_vote(wav_list, y_pred):
    # Create a dictionary to group segments by their common part
    segments_by_wav = {}
    for wav, pred in zip(wav_list, y_pred):
        common_part = wav.rsplit('_', 1)[0]  
        if common_part not in segments_by_wav:
            segments_by_wav[common_part] = []
        segments_by_wav[common_part].append(pred)

    # Perform majority vote for each group and get the final predictions
    final_predictions = []
    for common_part, segment_preds in segments_by_wav.items():
        counts = {pred: segment_preds.count(pred) for pred in set(segment_preds)}
        majority_prediction = max(counts, key=counts.get)
        final_predictions.append(majority_prediction)

    return final_predictions


def custom_folds_train_on_segments(json_folds, data_path, segment_size, test_seg, output_path, num_classes):
    try:
        with open(json_folds) as f:
            folds = json.load(f)
    except ValueError:
        print(f"{json_folds} is not a valid JSON file.")

    songs = crawl_directory(data_path, extension=".wav")

    scaler = StandardScaler()
    f1_macro_scorer = make_scorer(f1_score, average='macro')
    param_grid = {
        'C': [0.1, 1, 10, 50, 100, 1000], 
        'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10],
        'kernel': ['rbf'],
    }
    
    aggregated_cm = np.zeros((num_classes, num_classes), dtype=int)

    acc_scores = []
    f1_scores = []
    
    if os.path.exists(f'{json_folds.replace(".json", "")}_results.txt'):
        os.remove(f'{json_folds.replace(".json", "")}_results.txt')
    
    i=-1
    pattern = r'class_(\d+)'
    for fold in folds:
        print(f"\n========================= FOLD {i+1}  =========================\n")
        train_wavs, test_wavs = [], []
        model_name = 'classifier_' + fold

        # Split to train/test
        train_set = folds[fold]['train']
        test_set = folds[fold]['test']
        
        for song in songs:
            if os.path.basename(song) in train_set:
                train_wavs.append(os.path.basename(song))
            elif os.path.basename(song) in test_set:
                test_wavs.append(os.path.basename(song))

        print('Fold:', fold)
        print(f'{len(train_wavs)} songs in train set')
        print(f'{len(test_wavs)} songs in test set')
        
        print("\nTrimming and segmenting songs. This may take a while..")
        prepare_dirs(data_path, train_wavs, test_wavs, output_path, segment_size, test_seg)

        train_dirs = []
        train_dirs = get_subfolders(os.path.join(output_path, "train"))
        test_dir = os.path.join(output_path, "test")

        # train features
        features_list, class_names, file_names, shapes_list = feature_extractor(train_dirs, len(train_dirs))
        
        # create list of labels (labels as many as the shapes (from shapes_list))
        features_list = np.array(features_list)
        if (features_list.shape[0]) > 0:
            label_mapping = [class_mapping_dict[path.split('/')[-1]] for path in train_dirs]
            labels = []
            
            for count, label in zip(shapes_list, label_mapping):
                labels.extend([label]*count)
        else:
            raise ValueError("Features' list does not contain elements.")

        # test features & test labels
        mid_term_features, wav_file_list2, mid_feature_names = feature_extractor(test_dir, len(test_dir), train=False)
        y_labels = []
        for name in wav_file_list2:
            match = re.search(pattern, name)
            if match:
                class_number = int(match.group(1))
                y_labels.append(class_number)

        X_train = []
        y_train = []
        X_test = []
        y_test = []
        test_names = []

        fold_train_df = create_df(file_names, labels, features_list) 
        fold_train_df = fold_train_df.sample(frac=1).reset_index(drop=True)

        X_train.append(fold_train_df.iloc[:, 2:].values)
        y_train.append(fold_train_df['label'].values)

        fold_test_df = create_df(wav_file_list2, y_labels, mid_term_features)
        # fold_test_df = fold_test_df.sample(frac=1).reset_index(drop=True)

        test_names.append(fold_test_df['file_name'].values)
        test_names = test_names[0].tolist()

        X_test.append(fold_test_df.iloc[:, 2:].values)
        y_test.append(fold_test_df['label'].values)

        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        X_test = np.concatenate(X_test, axis=0)
        y_test = np.concatenate(y_test, axis=0)
        
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        svm = SVC()  
        grid_search = GridSearchCV(svm, param_grid, scoring=f1_macro_scorer)
        
        grid_search.fit(X_train, y_train)
        
        # best_model = grid_search.best_estimator_
        y_pred = grid_search.predict(X_test) 

        f1_macro_fold = grid_search.score(X_test, y_test)   

        y_pred = majority_vote(test_names, y_pred)    
        y_test = majority_vote(test_names, y_test)

        acc_fold = accuracy_score(y_test, y_pred)
        f1_fold = f1_score(y_test, y_pred, average = 'macro')
        fold_cm = confusion_matrix(y_test, y_pred)
        
        f1_scores.append(f1_fold)
        acc_scores.append(acc_fold)
        aggregated_cm = np.add(aggregated_cm, fold_cm)
        
        with open(f'{json_folds.replace(".json", "")}_results.txt', 'a') as f:
            print(f'\nBest parameters for split {i+1}: {grid_search.best_params_}')
            print(f"F1 (macro-averaged) for fold {i+1} (no majority): {f1_macro_fold}")
            print(f"Accuracy for fold {i+1}: {acc_fold}")        
            print(classification_report(y_test, y_pred))
            print(f"Confusion matrix: \n{fold_cm}")
            
            f.write(f"\n========================= FOLD {i+1}  =========================\n")
            f.write(f'\nBest parameters for split {i+1}: {grid_search.best_params_}')
            f.write(f"\nF1 (macro-averaged) for fold {i+1} (no majority): {f1_macro_fold}")
            f.write(f"\nAccuracy for fold {i+1}: {acc_fold}\n")        
            f.write(classification_report(y_test, y_pred))
            f.write(f"\nConfusion matrix: \n{fold_cm}")

        i += 1
        
        # Remove dirs for the next fold
        shutil.rmtree(os.path.join(output_path, "train"))
        shutil.rmtree(os.path.join(output_path, "test"))
        
    agg_f1_scores = round(np.mean(f1_scores)*100, 2)
    agg_std_f1_scores = round(np.std(f1_scores)*100, 2)
    
    agg_acc_scores = round(np.mean(acc_scores)*100, 2)
    agg_std_acc_scores = round(np.std(acc_scores)*100, 2)
    
    with open(f'{json_folds.replace(".json", "")}_results.txt', 'a') as f:
        print(f"\n#################### AGGREGATED RESULTS ####################")
        print(f"Aggregated f1-macro score ({len(folds)} folds): {agg_f1_scores}% with std: {agg_std_f1_scores}")
        print(f"Aggregated accuracy score ({len(folds)} folds): {agg_acc_scores}% with std: {agg_std_acc_scores}")
        print(f"Confusion matrix: \n{aggregated_cm}")
        
        f.write(f"\n#################### AGGREGATED RESULTS ####################")
        f.write(f"\nAggregated f1-macro score ({len(folds)} folds): {agg_f1_scores}% with std: {agg_std_f1_scores}")
        f.write(f"\nAggregated accuracy score ({len(folds)} folds): {agg_acc_scores}% with std: {agg_std_acc_scores}")
        f.write(f"\nConfusion matrix: \n{aggregated_cm}")
        
    return aggregated_cm


def kfold_cross_val(file_names, labels, features_list, fold):
    """
    Perform kfold cross validation.
    
    Args:
        file_names (_list_): the wav names
        labels (_list_): the labels to be used
        features_list (_list_): the feature vectors
        fold (_int_): num of folds
    """
    
    X = np.array(features_list)
    y = np.array(labels)
    
    scaler = StandardScaler()
    f1_macro_scorer = make_scorer(f1_score, average='macro')
    param_grid = {'C': [0.1, 1, 10, 50, 100, 1000], 
                  'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10],
                  'kernel': ['rbf'],
                 }
    
    aggregated_cm = np.zeros((9, 9), dtype=int)
    kfold = StratifiedKFold(n_splits=fold, shuffle=True, random_state=42)
    
    acc_scores = []
    f1_scores = []
    
    if os.path.exists(f'{fold}_fold_results.txt'):
        os.remove(f'{fold}_fold_results.txt')
    
    i = 0
    for train_index, test_index in kfold.split(X, y):
        print(f"\n========================= FOLD {i+1}  =========================\n")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        svm = SVC()
        grid_search = GridSearchCV(svm, param_grid=param_grid, cv=kfold, scoring=f1_macro_scorer)
        
        # perform grid search on training data
        grid_search.fit(X_train_scaled, y_train)
        y_pred = grid_search.predict(X_test_scaled)
        
        f1_macro_fold = grid_search.score(X_test_scaled, y_test)
        acc_fold = accuracy_score(y_test, y_pred)
        fold_cm = confusion_matrix(y_test, y_pred)
        
        f1_scores.append(f1_macro_fold)
        acc_scores.append(acc_fold)
        aggregated_cm = np.add(aggregated_cm, fold_cm)
        
        with open(f'{fold}_fold_results.txt', 'a') as f:
            print(f'\nBest parameters for split {i+1}: {grid_search.best_params_}')
            print(f"F1 (macro-averaged) for fold {i+1}: {f1_macro_fold}")
            print(f"Accuracy for fold {i+1}: {acc_fold}")        
            print(classification_report(y_test, y_pred))
            print(f"Confusion matrix: \n{fold_cm}")
            
            f.write(f"\n========================= FOLD {i+1}  =========================\n")
            f.write(f'\nBest parameters for split {i+1}: {grid_search.best_params_}')
            f.write(f"\nF1 (macro-averaged) for fold {i+1}: {f1_macro_fold}")
            f.write(f"\nAccuracy for fold {i+1}: {acc_fold}\n")        
            f.write(classification_report(y_test, y_pred))
            f.write(f"\nConfusion matrix: \n{fold_cm}")
        
        i+=1
    
    print(f"\n################## AGGREGATED RESULTS ##################\n")
    
    agg_f1_scores = round(np.mean(f1_scores)*100, 2)
    agg_std_f1_scores = round(np.std(f1_scores)*100, 2)
    
    agg_acc_scores = round(np.mean(acc_scores)*100, 2)
    agg_std_acc_scores = round(np.std(acc_scores)*100, 2)

    with open(f'{fold}_fold_results.txt', 'a') as f:
        print(f"\n#################### AGGREGATED RESULTS ####################")
        print(f"Aggregated f1-macro score ({fold} folds): {agg_f1_scores}% with std: {agg_std_f1_scores}")
        print(f"Aggregated accuracy score ({fold} folds): {agg_acc_scores}% with std: {agg_std_acc_scores}")
        print(f"Confusion matrix: \n{aggregated_cm}")
        
        f.write(f"\n#################### AGGREGATED RESULTS ####################")
        f.write(f"\nAggregated f1-macro score ({fold} folds): {agg_f1_scores}% with std: {agg_std_f1_scores}")
        f.write(f"\nAggregated accuracy score ({fold} folds): {agg_acc_scores}% with std: {agg_std_acc_scores}")
        f.write(f"\nConfusion matrix: \n{aggregated_cm}")

    return aggregated_cm