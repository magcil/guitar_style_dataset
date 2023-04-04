from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np


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
    
    param_grid = {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1,]}

    svm_model = SVC(kernel='rbf')
    kf = KFold(n_splits=fold, shuffle=True)
    grid_search = GridSearchCV(svm_model, param_grid, cv=kf)

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    print("Best score:", grid_search.best_score_)
    print(f"best model: {best_model}")
    
    # Predictions
    
    preds = best_model.predict(X_test)
        
    print(f"Test score: {best_model.score(X_test, y_test)}")
    print(classification_report(y_test, preds))
