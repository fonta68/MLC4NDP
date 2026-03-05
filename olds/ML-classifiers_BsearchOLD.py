##################### IMPORT ###############
import os
import numpy as np
import pandas as pd
#import tensorflow as tf
import argparse
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
#import matplotlib.pyplot as plt

# Import per classifcatori
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
# from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, f1_score, classification_report

from sklearn.metrics import (
    confusion_matrix,
    recall_score,
    roc_auc_score,
    cohen_kappa_score,
    classification_report
    roc_auc_score
)

from sklearn.preprocessing import label_binarize
import random
#################################################


if __name__ == '__main__':
    # Parsing degli argomenti (senza epochs e batch_size)
    parser = argparse.ArgumentParser(description="Classification algorithms")
    parser.add_argument('--infile', type=str, required=True, help="Nome del file CSV di input")
    parser.add_argument('--cls', type=str, required=True, help="Nome del file CSV di input")
    #parser.add_argument('--output_dir', type=str, required=True, help="Cartella di output per salvare file e grafici")
    
    args = parser.parse_args()
    name = args.cls

	# Si estrae il nome del file
    # f_name = os.path.splitext(os.path.basename(args.input_file))[0]
    
    
    #os.makedirs(args.output_dir, exist_ok=True)
    #os.makedirs(f_name, exist_ok=True)

    # Carica i dati
    data = pd.read_csv(args.input_file)
    
    
    seed = random.randint(0, 2**32 - 1)
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    
    #data.iloc[:, 1:] = data.iloc[:, 1:].applymap(lambda x: round(x, 4))
        
    # Divisione in training (80%) e test (20%)
    split_index = int(len(data) * 0.8)
    df_train_data = data.iloc[:split_index]
    df_test_data = data.iloc[split_index:]

    # df_train, df_test = get_fold(data, args.k_fold, 5)


    # from pandas 2 numpy
    train_data = df_train_data.to_numpy()
    test_data = df_test_data.to_numpy()

        
    # Lista per il summary finale: accumulerà le metriche per ogni metodo e per ogni orizzonte
    final_summary_list = []


    #############################################
    # BAYESIAN SEARCH 
    #############################################

    # Definizione dei regressori (incluso Ridge)
    match name:
    case "LRC":
        cls = LogisticRegression()
    case "RgC":
        cls = RidgeClassifier()
    case "RFC":
        cls = RandomForestClassifier(random_state=42)
    case "SVC":
        cls = SVC()
    case "GBC":
        cls = GradientBoostingClassifier(random_state=42)
    case "ETC":
        cls = ExtraTreesClassifier(random_state=42)
    case "KNC":
        cls = KNeighborsClassifier()
    case _:
        raise ValueError(f"Unknown classifier: {name}")

    
    # Definizione dei dizionari degli iperparametri per la BAYES search
    search_spaces = {
        "LRC": {},  # No hyperparams to search for basic LinearRegression
        "RgC": {
            # Example of searching alpha over a log-scaled range
            "alpha": Real(1e-10, 10, prior='log-uniform')
        },
        "RFC": {
            # Example: integer range for n_estimators, plus a few discrete depths
            "n_estimators": Integer(100, 1000, prior='uniform'),
            "max_depth": Categorical([None, 5, 10, 20])
        },
        "SVC": {
            # Because you're using MultiOutputClassifier(SVR(...)),
            # prefix the parameter names with 'estimator__'
            "C": Real(1e-4, 1e2, prior='log-uniform'),
            "epsilon": Real(1e-4, 10, prior='log-uniform')
        },
        "GBC": {
            "n_estimators": Integer(100, 1000, prior='uniform'),
            "learning_rate": Real(1e-3, 0.5, prior='log-uniform')
        },
        "ETC": {
            "n_estimators": Integer(100, 1000, prior='uniform'),
            "max_depth": Categorical([None, 5, 10, 20])
        },
        "KNC": {
            "n_neighbors": Integer(1, 15, prior='uniform')
        }
    }
    
    
    print("Bayesian search for {name}")
    if search_spaces[name]:  
        # If there are hyperparameters to tune:
        bayes_search = BayesSearchCV(
            estimator=cls,
            search_spaces=search_spaces[name],
            n_iter=25,  # number of parameter sets to try
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=3,
            random_state=42
        )
        bayes_search.fit(X_train, y_train)
        best_cls = bayes_search.best_estimator_
        f_bestpars.write(f"Best parameters for {name}: {bayes_search.best_params_}\n")
    else:
        # If no hyperparameters to tune:
        cls.fit(X_train, y_train)
        best_cls = cls

    # Predict the test labels
    preds_cls = best_cls.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, preds_cls)
    
    # F1 Score (default is binary; use average='weighted' for multi-class)
    f1 = f1_score(y_test, preds_cls, average='weighted')
    
    # Test predictions 
    y_preds = best_cls.predict(X_test)

    
    
    sen = recall_score(y_test, y_preds, average='macro')
    
    labels = np.unique(y_test)
    cm = confusion_matrix(y_test, y_preds, labels=labels)

    specificity_list = []

    for i in range(len(labels)):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fn + fp)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_list.append(specificity)

    spe = np.mean(specificity_list)

    y_test_bin = label_binarize(y_test, classes=labels)

    if hasattr(best_cls, "preds_probs"):
        y_proba = best_cls.preds_probs(X_test)
        auc = roc_auc_score(y_test_bin, y_probs, average='macro', multi_class='ovr')
    else:
        auc = None

    kappa = cohen_kappa_score(y_test, y_pred)

    print(f"{acc:.4f}\t{sen:.4f}\t{spe:.4f}\t{f1:.4f}\t{auc:.4f}\t{kappa:.4f}")

