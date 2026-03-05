# Environment For `ML-classifiers_BsearchBIN.py`

The Conda definition is in `environment-ML-classifiers_BsearchBIN.yml`.

Use Python 3.12. The script imports `catboost`, and that is a safer target than the local system `python3` (`3.13.5`).

## Plain `venv` alternative

If `conda` is not installed, use a standard virtual environment instead.

For sklearn-only classifiers (`SVC`, `RFC`, `KNC`, `LRC`, `RgC`, `GBC`, `ETC`):

```bash
python3 -m venv .venv-ml-classifiers-lite
source .venv-ml-classifiers-lite/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-ML-classifiers_BsearchBIN-lite.txt
```

For the full environment (includes `lightgbm` and `catboost`):

```bash
python3 -m venv .venv-ml-classifiers
source .venv-ml-classifiers/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-ML-classifiers_BsearchBIN.txt
```

Note: the local `python3` is `3.13.5`. `catboost` and sometimes `lightgbm` can be slower to support the newest Python release, so the full install may fail on 3.13 even though the sklearn-only one should be fine.

## Create the Conda environment

```bash
conda env create -f environment-ML-classifiers_BsearchBIN.yml
conda activate ml-classifiers-bsearchbin
```

If the environment already exists and you update the file:

```bash
conda env update -f environment-ML-classifiers_BsearchBIN.yml --prune
```

## Run the script

```bash
conda activate ml-classifiers-bsearchbin
python ML-classifiers_BsearchBIN.py --infile your_data.csv --cls SVC --labels ADD DLB
```

## Lightweight environment

If you only use classifiers that do not require `CatBoostClassifier` (for example `SVC`, `RFC`, `KNC`, `LRC`, `RgC`, `GBC`, `ETC`, `LGBM`), you can use the lighter file:

```bash
conda env create -f environment-ML-classifiers_BsearchBIN-lite.yml
conda activate ml-classifiers-bsearchbin-lite
```

## Packages included

- `python=3.12`
- `numpy`
- `pandas`
- `scikit-learn`
- `scikit-optimize`
- `lightgbm`
- `catboost`
