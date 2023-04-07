from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import os
import sys
import json
import matplotlib.pyplot as plt

def plot_cm(conf_matrix, class_names):

    # Create a figure object and add a subplot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    # Plot the confusion matrix as an image
    im = ax.imshow(conf_matrix, cmap='Blues')

    # Add annotations to the image
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, conf_matrix[i, j], ha='center', va='center', fontsize=8)

    # Add a colorbar legend to the plot
    fig.colorbar(im)

    # Set the tick labels for the x-axis and y-axis
    ax.set_xticklabels([''] + class_names, fontsize=8, rotation=45, ha='right')
    ax.set_yticklabels([''] + class_names, fontsize=8)

    # Set the labels for the x-axis and y-axis
    plt.xlabel('Predicted', fontsize=10)
    plt.ylabel('True', fontsize=10)

    # Add a title to the plot
    plt.title('Confusion Matrix', fontsize=12)

    # Save the plot as an EPS file
    plt.savefig(f'{len(class_names)}_class_confusion_matrix.eps', format='eps')
    


def kfold_cross_val(features_list, file_names, labels, fold):
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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42, 
                                                        stratify=labels)
    
    # scaling
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"X_train shape: {X_train.shape} \nX_test shape: {X_test.shape}")
    
    param_grid = {'C': [0.1, 1, 10, 20, 40, 50, 100], 
                  'gamma': [0.001, 0.01, 0.1, 1, 10],
                  'kernel': ['linear', 'rbf' ]
                  }

    svm_model = SVC()
    kf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=42)
    grid_search = GridSearchCV(svm_model, param_grid, cv=kf, verbose=3, scoring='f1_macro')

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    print("Best score:", grid_search.best_score_)
    print(f"best model: {best_model}")
    
    # Predictions
    preds = best_model.predict(X_test)
        
    print(f"Test score: {best_model.score(X_test, y_test)}")
    print(classification_report(y_test, preds))
    
    cm = confusion_matrix(y_test, preds)
    print(f"Confusion matrix: \n{cm}")
    

def leave_one_metadata_out(df: pd.DataFrame, fold):
    unique_metadata = df[fold].unique().tolist()
    
    scaler = StandardScaler()
    print(f"{fold}s: {unique_metadata}")
    
    for idx, metadata in enumerate(unique_metadata):
        train_metadata = unique_metadata.copy()
        train_metadata.remove(metadata)
        print(f"{fold}s for {idx} training: {train_metadata}")
        print(f"{fold} for {idx} test: {[metadata]}")

        X_train = df[df[fold].isin(train_metadata)].iloc[:, 2:138]
        y_train = df[df[fold].isin(train_metadata)]['label']
        
        X_test = df[df[fold].isin([metadata])].iloc[:, 2:138]
        y_test = df[df[fold].isin([metadata])]['label']
        
        print(f"\n X_train: {len(X_train)} y_train: {len(y_train)}"+
            f"\n X_test: {len(X_test)} y_test: {len(y_test)}\n")
    
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    

def ready_folds_train(file_names, labels, features_list, ready_folds):
        
    try:
        with open(ready_folds) as f:
            folds = json.load(f)
    except ValueError:
        print(f"{ready_folds} is not a valid JSON file.")
        
    # 1st col: wav_names, 2nd col: labels, the rest cols represent the features
    file_names = [os.path.basename(wav_name) for wav_name in file_names]
    df = pd.DataFrame({
        'file_name': file_names,
        'label': labels
    })
    
    features_list = pd.DataFrame(features_list.tolist())
    df = pd.concat([df, features_list], axis=1)
    
    scaler = StandardScaler()
    param_grid = {'C': [0.1, 1, 10, 20, 40, 50, 100], 
                'gamma': [0.001, 0.01, 0.1, 1, 10],
                'kernel': ['linear', 'rbf' ]
                }
        
    svm = SVC()  
    grid_search = GridSearchCV(svm, param_grid, scoring='f1_macro')
    aggregated_cm = np.zeros((9, 9), dtype=int)
    aggregated_score = 0.0
    
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
        
        print(f"X_train_shape: {X_train.shape} y_train_shape: {y_train.shape}")
        print(f"X_test_shape: {X_test.shape} y_test_shape: {y_test.shape}")

        grid_search.fit(X_train, y_train)
        
        print(f'\nBest parameters for split {i}: {grid_search.best_params_}')

        best_model = grid_search.best_estimator_
        preds = best_model.predict(X_test)
        
        test_score = best_model.score(X_test, y_test)
        print(f"Test score: {test_score}")
        
        aggregated_score += test_score
        
        print(classification_report(y_test, preds))
        
        fold_cm = confusion_matrix(y_test, preds)
        aggregated_cm = np.add(aggregated_cm, fold_cm)
        print(f"Confusion matrix: \n{fold_cm}")
    
    print(f"\n################## AGGREGATED RESULTS ##################")
    
    aggregated_score = round(aggregated_score / len(folds)*100,2)
    print(f"Aggregated test accuracy ({len(folds)} folds): {aggregated_score}%\n")
    print(aggregated_cm)

    return aggregated_cm