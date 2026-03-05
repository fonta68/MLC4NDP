# MLC4NDP

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Machine Learning Classifiers for Neurodegenerative Prediction.

This repository contains Python scripts to run Bayesian hyperparameter search and evaluate multiple classifiers on neurodegenerative datasets.

## What is included

- Binary classification workflow with label filtering and repeated runs.
- Bayesian optimization (`BayesSearchCV`) for model hyperparameters.
- Metrics: Accuracy, Sensitivity (Recall), Specificity, F1, AUC, Cohen's Kappa.
- Optional LightGBM and CatBoost support.

Main scripts:

- `ML-classifiers_BsearchBINClaude.py`: recommended binary pipeline.
- `ML-classifiers_BsearchBIN.py`: alternative binary pipeline.
- `ML-classifiers_Bsearch.py`: legacy/general classifier script.
- `ML-classifiers_BsearchBin.sh`: helper script for repeated runs.
- `setup-venv-ML-classifiers_BsearchBIN.sh`: virtual environment bootstrap.

## Requirements

- Python 3.9+ (recommended)
- pip

Install dependencies (full set):

```bash
python3 -m venv .venv-ml-classifiers
source .venv-ml-classifiers/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-ML-classifiers_BsearchBIN.txt
```

Lite dependencies (without LightGBM/CatBoost):

```bash
python -m pip install -r requirements-ML-classifiers_BsearchBIN-lite.txt
```

You can also use the bootstrap script:

```bash
bash setup-venv-ML-classifiers_BsearchBIN.sh
```

## Input data format

Input must be a CSV file containing:

- A target column named `class`
- A feature column named `Sex` (used explicitly in scaling logic)
- Other numeric feature columns

## Run examples

Run the recommended binary pipeline directly:

```bash
python3 ML-classifiers_BsearchBINClaude.py \
  --infile your_data.csv \
  --cls LRC \
  --labels ADD DLB
```

Run repeated experiments via shell helper:

```bash
bash ML-classifiers_BsearchBin.sh LRC your_data.csv 20 "ADD DLB"
```

## Supported classifier codes

- `LRC`: LogisticRegression
- `RgC`: RidgeClassifier
- `RFC`: RandomForestClassifier
- `SVC`: Support Vector Classifier
- `GBC`: GradientBoostingClassifier
- `ETC`: ExtraTreesClassifier
- `KNC`: KNeighborsClassifier
- `LGC`: LightGBMClassifier
- `CBC`: CatBoostClassifier

## Output

Scripts print a line starting with `RES:` containing the main evaluation metrics. The shell wrapper writes run outputs to result text files.

## Repository notes

To keep the repository lightweight, local datasets/results and some local experiment folders are ignored in `.gitignore`:

- `*.csv`, `*.xlsx`
- `*_res.txt`, `*.res.txt`
- `giovi/`, `olds/`, `catboost_info/`

If you need to share data, use external storage or Git LFS.
