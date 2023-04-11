from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, make_scorer, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC, LinearSVC
import numpy as np
import pandas as pd
import os
import sys
import json
import matplotlib.pyplot as plt


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
            plt.title('Guitar-Aggregated Confusion Matrix', fontsize=12)
            plt.savefig(f'guitars_{len(class_names)}_class_confusion_matrix.eps', format='eps')
        elif 'amplifier' in folds.lower():
            plt.title('Amplifier-Aggregated Confusion Matrix', fontsize=12)
            plt.savefig(f'amplifiers_{len(class_names)}_class_confusion_matrix.eps', format='eps')
    else:
        plt.title('Aggregated Confusion Matrix', fontsize=12)
        # Save the plot as an EPS file
        plt.savefig(f'{len(class_names)}_class_confusion_matrix.eps', format='eps')


def kfold_cross_val(file_names, labels, features_list, fold):
    """
    Perform cross validation on the given features.
    
    Args:
        features_list (_list_): the feature vectors
        file_names (_list_): the wav names
        labels (_list_): the labels to be used
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
        
        print(f"F1 (macro-averaged) for fold {i+1}: {f1_macro_fold}")
        print(f"Accuracy for fold {i+1}: {acc_fold}")        
        print(classification_report(y_test, y_pred))
        print(f"Confusion matrix: \n{fold_cm}")
        i+=1
    
    print(f"\n################## AGGREGATED RESULTS ##################\n")
    
    agg_f1_scores = round(np.mean(f1_scores)*100, 2)
    agg_std_f1_scores = round(np.std(f1_scores)*100, 2)
    
    agg_acc_scores = round(np.mean(acc_scores)*100, 2)
    agg_std_acc_scores = round(np.std(acc_scores)*100, 2)
    
    print(f"Aggregated f1-macro score ({fold} folds): {agg_f1_scores}% with std: {agg_std_f1_scores}\n")
    print(f"Aggregated accuracy score ({fold} folds): {agg_acc_scores}% with std: {agg_std_acc_scores}\n")
    print(aggregated_cm)

    return aggregated_cm
    
    
    # scaling
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    
    # print(f"X_train shape: {X_train.shape} \nX_test shape: {X_test.shape}")
    
    # param_grid = {'C': [0.1, 1, 10, 20, 40, 50, 100], 
    #               'gamma': [0.001, 0.01, 0.1, 1, 10],
    #               'kernel': ['linear', 'rbf' ]
    #               }

    # svm_model = SVC()
    # kf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=42)
    # grid_search = GridSearchCV(svm_model, param_grid, cv=kf, scoring='f1_macro')

    # grid_search.fit(X_train, y_train)
    # best_model = grid_search.best_estimator_
    
    # print("Best score:", grid_search.best_score_)
    # print(f"best model: {best_model}")
    
    # # Predictions
    # preds = best_model.predict(X_test)
        
    # print(f"Test score: {best_model.score(X_test, y_test)}")
    # print(classification_report(y_test, preds))
    
    # cm = confusion_matrix(y_test, preds)
    # print(f"Confusion matrix: \n{cm}")
    

def create_df(file_names, labels, features_list):
    file_names = [os.path.basename(wav_name) for wav_name in file_names]
    df = pd.DataFrame({
        'file_name': file_names,
        'label': labels
    })
    features_list = pd.DataFrame(features_list.tolist())
    df = pd.concat([df, features_list], axis=1)
    
    return df


def leave_one_metadata_out(file_names, labels, features_list, metada_folds):
    try:
        with open(metada_folds) as f:
            mt_folds = json.load(f)
    except ValueError:
        print(f"{metada_folds} is not a valid JSON file.")

    df = create_df(file_names, labels, features_list) 
    
    scaler = StandardScaler()
    f1_macro_scorer = make_scorer(f1_score, average='macro')
    param_grid = {'C': [0.1, 1, 10, 50, 100, 1000], 
                  'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10],
                  'kernel': ['rbf'],
                 }
    
    aggregated_cm = np.zeros((9, 9), dtype=int)

    acc_scores = []
    f1_scores = []
    
    for i, key in enumerate(mt_folds.keys()):
        print(f"\n========================= FOLD {i+1}  =========================\n")
        
        metadata_list = mt_folds[key]
                
        mask = df['file_name'].isin(metadata_list)
        
        X_test = df[mask].iloc[:, 2:].values
        y_test = df[mask]['label'].values
        X_train = df[~mask].iloc[:, 2:].values
        y_train = df[~mask]['label'].values
        
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
                
        print(f"X_train_shape: {X_train.shape} y_train_shape: {y_train.shape}")
        print(f"X_test_shape: {X_test.shape} y_test_shape: {y_test.shape}")
        
        svm = SVC(random_state=40)  
        grid_search = GridSearchCV(svm, param_grid, scoring=f1_macro_scorer)
        
        grid_search.fit(X_train, y_train)
        print(f'\nBest parameters for split {i+1}: {grid_search.best_params_}')
        
        # best_model = grid_search.best_estimator_
        y_pred = grid_search.predict(X_test)        
        
        f1_macro_fold = grid_search.score(X_test, y_test)
        acc_fold = accuracy_score(y_test, y_pred)
        fold_cm = confusion_matrix(y_test, y_pred)
        
        f1_scores.append(f1_macro_fold)
        acc_scores.append(acc_fold)
        aggregated_cm = np.add(aggregated_cm, fold_cm)
        
        print(f"F1 (macro-averaged) for fold {i+1}: {f1_macro_fold}")
        print(f"Accuracy for fold {i+1}: {acc_fold}")        
        print(classification_report(y_test, y_pred))
        print(f"Confusion matrix: \n{fold_cm}")
    
    print(f"\n################## AGGREGATED RESULTS ##################\n")
    
    agg_f1_scores = round(np.mean(f1_scores)*100, 2)
    agg_std_f1_scores = round(np.std(f1_scores)*100, 2)
    
    agg_acc_scores = round(np.mean(acc_scores)*100, 2)
    agg_std_acc_scores = round(np.std(acc_scores)*100, 2)
    
    print(f"Aggregated f1-macro score ({len(mt_folds)} folds): {agg_f1_scores}% with std: {agg_std_f1_scores}\n")
    print(f"Aggregated accuracy score ({len(mt_folds)} folds): {agg_acc_scores}% with std: {agg_std_acc_scores}\n")
    print(aggregated_cm)

    return aggregated_cm


def ready_folds_train(file_names, labels, features_list, ready_folds):
    try:
        with open(ready_folds) as f:
            folds = json.load(f)
    except ValueError:
        print(f"{ready_folds} is not a valid JSON file.")

    df = create_df(file_names, labels, features_list)   
     
    scaler = StandardScaler()
    f1_macro_scorer = make_scorer(f1_score, average='macro')
    param_grid = {'C': [0.1, 1, 10, 50, 100, 1000], 
                  'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10],
                  'kernel': ['rbf'],
                 }
    
    aggregated_cm = np.zeros((9, 9), dtype=int)

    acc_scores = []
    f1_scores = []

    for i in range(len(folds)):
        print(f"\n========================= FOLD {i+1}  =========================\n")
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        train_files = folds[f'fold_{i}']['train']
        test_files = folds[f'fold_{i}']['test']
        
        # Filter the dataframe to only include the train files for this fold & shuffle
        fold_train_df = df[df['file_name'].isin(train_files)]
        fold_train_df = fold_train_df.sample(frac=1).reset_index(drop=True)
        
        X_train.append(fold_train_df.iloc[:, 2:].values)
        y_train.append(fold_train_df['label'].values)
        
        # Filter the dataframe to only include the test files for this fold & shuffle
        fold_test_df = df[df['file_name'].isin(test_files)]
        fold_test_df = fold_test_df.sample(frac=1).reset_index(drop=True)
        
        X_test.append(fold_test_df.iloc[:, 2:].values)
        y_test.append(fold_test_df['label'].values)
        
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        X_test = np.concatenate(X_test, axis=0)
        y_test = np.concatenate(y_test, axis=0)
        
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        svm = SVC(random_state=40)  
        grid_search = GridSearchCV(svm, param_grid, scoring=f1_macro_scorer)
        
        grid_search.fit(X_train, y_train)
        print(f'\nBest parameters for split {i+1}: {grid_search.best_params_}')
        
        # best_model = grid_search.best_estimator_
        y_pred = grid_search.predict(X_test)        
        
        f1_macro_fold = grid_search.score(X_test, y_test)
        acc_fold = accuracy_score(y_test, y_pred)
        fold_cm = confusion_matrix(y_test, y_pred)
        
        f1_scores.append(f1_macro_fold)
        acc_scores.append(acc_fold)
        aggregated_cm = np.add(aggregated_cm, fold_cm)
        
        print(f"F1 (macro-averaged) for fold {i+1}: {f1_macro_fold}")
        print(f"Accuracy for fold {i+1}: {acc_fold}")        
        print(classification_report(y_test, y_pred))
        print(f"Confusion matrix: \n{fold_cm}")
    
    print(f"\n################## AGGREGATED RESULTS ##################\n")
    
    agg_f1_scores = round(np.mean(f1_scores)*100, 2)
    agg_std_f1_scores = round(np.std(f1_scores)*100, 2)
    
    agg_acc_scores = round(np.mean(acc_scores)*100, 2)
    agg_std_acc_scores = round(np.std(acc_scores)*100, 2)
    
    print(f"Aggregated f1-macro score ({len(folds)} folds): {agg_f1_scores}% with std: {agg_std_f1_scores}\n")
    print(f"Aggregated accuracy score ({len(folds)} folds): {agg_acc_scores}% with std: {agg_std_acc_scores}\n")
    print(aggregated_cm)

    return aggregated_cm
