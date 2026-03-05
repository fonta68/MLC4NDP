#!/bin/bash

set -e

python3 -m venv .venv-ml-classifiers
source .venv-ml-classifiers/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-ML-classifiers_BsearchBIN.txt
