# iris-flower-classification

A small example project that trains a RandomForest classifier on the classic Iris dataset using scikit-learn.

## Install

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Train and save a model:

```bash
python iris_classifier.py --train
```

Predict one sample from the test set (loads saved model):

```bash
python iris_classifier.py --predict-sample
```

Run hyperparameter tuning (saves best model to `models/iris_rf_tuned.joblib`):

```bash
python tune.py
```

Explore the notebook `notebooks/iris_analysis.ipynb` for visualizations.

## Tests

Run the test suite with:

```bash
pytest
```

## Docker

Build and run the example Docker image:

```bash
docker build -t iris-classifier .
docker run --rm iris-classifier
```

