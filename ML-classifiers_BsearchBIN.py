##################### IMPORT ###############
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

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    recall_score,
    roc_auc_score,
    cohen_kappa_score,
)


from sklearn.preprocessing import label_binarize
import random
#################################################


if __name__ == '__main__':
    # Parsing degli argomenti (senza epochs e batch_size)
    parser = argparse.ArgumentParser(description="Classification algorithms")
    parser.add_argument('--infile', type=str, required=True, help="Nome del file CSV di input")
    #parser.add_argument('--cls', type=str, required=True, help="Nome\n del file CSV di input")
    #parser.add_argument('--output_dir', type=str, required=True, help="Cartella di output per salvare file e grafici")
    parser.add_argument(
        '--cls',
        type=str,
        required=True,
        help="""Classifier to use.
        Available options:
        - LRC: LogisticRegression
        - RgC: RidgeClassifier
        - RFC: RandomForestClassifier
        - SVC: Support Vector Classifier
        - GBC: GradientBoostingClassifier
        - ETC: ExtraTreesClassifier
        - KNC: KNeighborsClassifier
        - LGC: LightGBMClassifier
        - CBC: CatBoostClassifier"""
    )
    parser.add_argument('--labels', nargs='+', required=True, help='Etichette da selezionare (es: ADD DLB)')
    
    args = parser.parse_args()
    name = args.cls
    
    # Lista delle etichette selezionate
    selected_classes = args.labels

    # Carica i dati
    data = pd.read_csv(args.infile)
    
    
    # Filtra il DataFrame
    data = data[data['class'].isin(selected_classes)]
    
    
    # Stampa il risultato (o salvalo, ecc.)
    #print("Classi selezionate:", selected_classes)
    

    
        
    seed = random.randint(0, 2**32 - 1)
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    #data.to_csv("F.csv")
    #data.iloc[:, 1:] = data.iloc[:, 1:].applymap(lambda x: round(x, 4))
        
    # Divisione in training (80%) e test (20%)
    split_index = int(len(data) * 0.8)
    df_train_data = data.iloc[:split_index]
    df_test_data = data.iloc[split_index:]

    # df_train, df_test = get_fold(data, args.k_fold, 5)
    
    # df_train_data.to_csv("train_data.csv", index=False)
    # df_test_data.to_csv("test_data.csv", index=False)
    
     
    # from pandas 2 numpy
    # train_data = df_train_data.to_numpy()
    # test_data = df_test_data.to_numpy()
    
    
    X_train = df_train_data.drop(columns='class')
    y_train = df_train_data['class']

    X_test = df_test_data.drop(columns='class')
    y_test = df_test_data['class']
    
    print("X_train shape:", X_train.shape)
    
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler()

    # Models that need scaling
    models_requiring_scaling = ["SVC", "KNC", "LRC", "RgC"]

    if name in models_requiring_scaling:
        # Separate 'Sex' to preserve it and its position
        sex_train = X_train[['Sex']]
        sex_test = X_test[['Sex']]

        features_to_scale = X_train.columns.drop('Sex')

        X_train_scaled = scaler.fit_transform(X_train[features_to_scale])
        X_test_scaled = scaler.transform(X_test[features_to_scale])

        # Rebuild DataFrames with 'Sex' first
        X_train = pd.concat([sex_train, pd.DataFrame(X_train_scaled, columns=features_to_scale, index=X_train.index)], axis=1)
        X_test  = pd.concat([sex_test, pd.DataFrame(X_test_scaled, columns=features_to_scale, index=X_test.index)], axis=1)

        
    


    #############################################
    # BAYESIAN SEARCH 
    #############################################
    '''

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
        
    '''
    if name == "LRC":
        cls = LogisticRegression()
    elif name == "RgC":
        cls = RidgeClassifier()
    elif name == "RFC":
        cls = RandomForestClassifier(random_state=42)
    elif name == "SVC":
        cls = SVC()
    elif name == "GBC":
        cls = GradientBoostingClassifier(random_state=42)
    elif name == "ETC":
        cls = ExtraTreesClassifier(random_state=42)
    elif name == "KNC":
        cls = KNeighborsClassifier()
    elif name == "LGC":
        from lightgbm import LGBMClassifier
        cls = LGBMClassifier(random_state=42, verbosity=-1)
    elif name == "CBC":
        from catboost import CatBoostClassifier
        cls = CatBoostClassifier(verbose=0, random_state=42)
    else:
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
            "C": Real(1e-4, 1e2, prior='log-uniform'),
            "gamma": Real(1e-4, 1e1, prior='log-uniform')
            #"kernel": Categorical(['rbf'])
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
        },
        "LGC": {
            "n_estimators": Integer(100, 1000, prior='uniform'),
            "learning_rate": Real(1e-3, 0.3, prior='log-uniform'),
            "num_leaves": Integer(15, 255, prior='uniform'),
            "max_depth": Categorical([-1, 3, 5, 10, 20]),
            "subsample": Real(0.5, 1.0, prior='uniform'),
            "colsample_bytree": Real(0.5, 1.0, prior='uniform')
        },
        "CBC": {
        "iterations": Integer(100, 1000),
        "learning_rate": Real(0.01, 0.3, prior='log-uniform'),
        "depth": Integer(3, 10),
        "l2_leaf_reg": Real(1, 10, prior='log-uniform')
        }
    }
    
    
    print(f"Bayesian search for {name}")
    if search_spaces[name]:  
        # If there are hyperparameters to tune:
        bayes_search = BayesSearchCV(
            estimator=cls,
            search_spaces=search_spaces[name],
            n_iter=25,  # number of parameter sets to try
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0,
            random_state=42
        )
        bayes_search.fit(X_train, y_train)
        best_cls = bayes_search.best_estimator_
        #f_bestpars.write(f"Best parameters for {name}: {bayes_search.best_params_}\n")
    else:
        # If no hyperparameters to tune:
        #print("NO")    
        cls.fit(X_train, y_train)
        best_cls = cls

    # Predict the test labels
    preds_cls = best_cls.predict(X_test)
    
    # TRAIN CHECK
    # preds_train = best_cls.predict(X_train)
    # acc_tr = accuracy_score(y_train, preds_train)
    #acc_ts = accuracy_score(y_test, preds_cls)
    #print(f"TRAIN:{acc_tr:.4f} TEST:{acc_ts:.4f}")
    


    
    from sklearn.preprocessing import label_binarize

    # Specify positive class
    positive_label = selected_classes[1]

    # Convert to binary: 1 for positive_label, 0 otherwise
    y_test_bin = [1 if y == positive_label else 0 for y in y_test]
    preds_bin = [1 if y == positive_label else 0 for y in preds_cls]
    y_preds_bin = [1 if y == positive_label else 0 for y in best_cls.predict(X_test)]

    # Accuracy
    acc = accuracy_score(y_test_bin, preds_bin)

    # F1 score
    f1 = f1_score(y_test_bin, preds_bin)

    # Sensitivity (Recall)
    sen = recall_score(y_test_bin, y_preds_bin)

    # Confusion matrix
    cm = confusion_matrix(y_test_bin, y_preds_bin)
    tn, fp, fn, tp = cm.ravel()
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0

    # ROC AUC (only if your model can give probabilities)
    if hasattr(best_cls, "preds_probs"):
        y_probs = best_cls.preds_probs(X_test)
        y_probs_bin = y_probs[:, 1] if y_probs.ndim > 1 else y_probs  # Second column = positive class prob
        auc = roc_auc_score(y_test_bin, y_probs_bin)
    else:
        auc = None

    # Cohen's kappa
    kappa = cohen_kappa_score(y_test_bin, y_preds_bin)

    # Print results
    print(f"RES:{acc:.4f}\t{sen:.4f}\t{spe:.4f}\t{f1:.4f}")
    # print(f"AUC: {auc:.4f}\tKappa: {kappa:.4f}")
