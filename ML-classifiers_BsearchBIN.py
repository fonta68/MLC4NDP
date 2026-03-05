##################### IMPORT ###############
import numpy as np
import pandas as pd
import argparse
import random

from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical

# Classifiers
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
    cohen_kappa_score,
    confusion_matrix,
)
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
#################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classification algorithms")
    parser.add_argument('--infile', type=str, required=True, help="Input CSV file")
    parser.add_argument(
        '--cls',
        type=str,
        required=True,
        help="""Classifier to use. Available options:
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
    parser.add_argument('--labels', nargs='+', required=True,
                        help='Two class labels to select (e.g.: ADD DLB). '
                             'The second label is treated as the positive class.')
    args = parser.parse_args()

    name = args.cls
    selected_classes = args.labels

    if len(selected_classes) != 2:
        raise ValueError("Exactly 2 labels must be provided via --labels.")

    # ── Load & filter ────────────────────────────────────────────────────────
    data = pd.read_csv(args.infile)
    data = data[data['class'].isin(selected_classes)]

    # ── Shuffle ──────────────────────────────────────────────────────────────
    seed = random.randint(0, 2**32 - 1)
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)

    # ── Stratified 80/20 split ───────────────────────────────────────────────
    X = data.drop(columns='class')
    y = data['class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    print("X_train shape:", X_train.shape)

    # ── Scaling (only for models that need it) ───────────────────────────────
    models_requiring_scaling = ["SVC", "KNC", "LRC", "RgC"]

    if name in models_requiring_scaling:
        sex_train = X_train[['Sex']]
        sex_test  = X_test[['Sex']]

        features_to_scale = X_train.columns.drop('Sex')

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train[features_to_scale])
        X_test_scaled  = scaler.transform(X_test[features_to_scale])

        X_train = pd.concat([
            sex_train.reset_index(drop=True),
            pd.DataFrame(X_train_scaled, columns=features_to_scale)
        ], axis=1)
        X_test = pd.concat([
            sex_test.reset_index(drop=True),
            pd.DataFrame(X_test_scaled, columns=features_to_scale)
        ], axis=1)

    # ── Classifier definitions ───────────────────────────────────────────────
    if name == "LRC":
        cls = LogisticRegression(max_iter=1000)
    elif name == "RgC":
        cls = RidgeClassifier()
    elif name == "RFC":
        cls = RandomForestClassifier(random_state=42)
    elif name == "SVC":
        cls = SVC(probability=True, random_state=42)   # probability=True enables predict_proba
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

    # ── Hyperparameter search spaces ─────────────────────────────────────────
    search_spaces = {
        "LRC": {
            "C": Real(1e-4, 1e2, prior='log-uniform'),
            "solver": Categorical(['lbfgs', 'saga']),
        },
        "RgC": {
            "alpha": Real(1e-10, 10, prior='log-uniform'),
        },
        "RFC": {
            "n_estimators": Integer(100, 1000, prior='uniform'),
            "max_depth": Categorical([None, 5, 10, 20]),
        },
        "SVC": {
            "C": Real(1e-4, 1e2, prior='log-uniform'),
            "gamma": Real(1e-4, 1e1, prior='log-uniform'),
        },
        "GBC": {
            "n_estimators": Integer(100, 1000, prior='uniform'),
            "learning_rate": Real(1e-3, 0.5, prior='log-uniform'),
        },
        "ETC": {
            "n_estimators": Integer(100, 1000, prior='uniform'),
            "max_depth": Categorical([None, 5, 10, 20]),
        },
        "KNC": {
            "n_neighbors": Integer(1, 15, prior='uniform'),
        },
        "LGC": {
            "n_estimators": Integer(100, 1000, prior='uniform'),
            "learning_rate": Real(1e-3, 0.3, prior='log-uniform'),
            "num_leaves": Integer(15, 255, prior='uniform'),
            "max_depth": Categorical([-1, 3, 5, 10, 20]),
            "subsample": Real(0.5, 1.0, prior='uniform'),
            "colsample_bytree": Real(0.5, 1.0, prior='uniform'),
        },
        "CBC": {
            "iterations": Integer(100, 1000),
            "learning_rate": Real(0.01, 0.3, prior='log-uniform'),
            "depth": Integer(3, 10),
            "l2_leaf_reg": Real(1, 10, prior='log-uniform'),
        },
    }

    # ── Bayesian search / fit ────────────────────────────────────────────────
    print(f"Bayesian search for {name}")
    if search_spaces[name]:
        bayes_search = BayesSearchCV(
            estimator=cls,
            search_spaces=search_spaces[name],
            n_iter=25,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0,
            random_state=42,
        )
        bayes_search.fit(X_train, y_train)
        best_cls = bayes_search.best_estimator_
    else:
        cls.fit(X_train, y_train)
        best_cls = cls

    # ── Predictions ──────────────────────────────────────────────────────────
    preds_cls = best_cls.predict(X_test)

    positive_label = selected_classes[1]
    y_test_bin = [1 if y == positive_label else 0 for y in y_test]
    preds_bin  = [1 if y == positive_label else 0 for y in preds_cls]

    # ── Metrics ──────────────────────────────────────────────────────────────
    acc   = accuracy_score(y_test_bin, preds_bin)
    f1    = f1_score(y_test_bin, preds_bin)
    sen   = recall_score(y_test_bin, preds_bin)
    kappa = cohen_kappa_score(y_test_bin, preds_bin)

    cm = confusion_matrix(y_test_bin, preds_bin)
    tn, fp, fn, tp = cm.ravel()
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0

    # AUC — three strategies depending on model capabilities
    if hasattr(best_cls, "predict_proba"):
        # Standard: use class probabilities
        y_probs = best_cls.predict_proba(X_test)
        y_probs_pos = y_probs[:, list(best_cls.classes_).index(positive_label)]
        auc = roc_auc_score(y_test_bin, y_probs_pos)
    elif hasattr(best_cls, "decision_function"):
        # RidgeClassifier and linear SVMs expose decision scores
        scores = best_cls.decision_function(X_test)
        auc = roc_auc_score(y_test_bin, scores)
    else:
        auc = None

    # ── Output ───────────────────────────────────────────────────────────────
    print(f"RES:{acc:.4f}\t{sen:.4f}\t{spe:.4f}\t{f1:.4f}", end="")
    if auc is not None:
        print(f"\tAUC:{auc:.4f}\tKappa:{kappa:.4f}")
    else:
        print(f"\tAUC:N/A\tKappa:{kappa:.4f}")
