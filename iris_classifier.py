"""Iris classifier utility module.

Provides functions to train, evaluate, save, and load a RandomForest
classifier for the Iris dataset and a small CLI for quick demos.
"""
from pathlib import Path
import argparse
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def load_data():
    iris = load_iris()
    return iris.data, iris.target, iris.target_names


def train_model(X_train, y_train, n_estimators=100, random_state=42, **kwargs):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, **kwargs)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, target_names=None):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)
    cm = confusion_matrix(y_test, y_pred)
    return {"accuracy": acc, "report": report, "confusion_matrix": cm}


def save_model(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path):
    return joblib.load(path)


def predict_sample(model, sample: np.ndarray, target_names=None):
    pred = model.predict(sample.reshape(1, -1))[0]
    return pred if target_names is None else target_names[pred]


def main():
    parser = argparse.ArgumentParser(description="Train or run the Iris classifier demo.")
    parser.add_argument("--train", action="store_true", help="Train and save a model")
    parser.add_argument("--model-path", type=Path, default=MODEL_DIR / "iris_rf.joblib", help="Where to save/load the model")
    parser.add_argument("--predict-sample", action="store_true", help="Load model and predict one sample from test set")
    parser.add_argument("--n-estimators", type=int, default=100)
    args = parser.parse_args()

    X, y, target_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    if args.train:
        model = train_model(X_train, y_train, n_estimators=args.n_estimators)
        save_model(model, args.model_path)
        print(f"Trained model saved to {args.model_path}")
        res = evaluate_model(model, X_test, y_test, target_names)
        print("Accuracy:", res["accuracy"])
        print("Classification report:\n", res["report"])

    if args.predict_sample:
        model = load_model(args.model_path)
        sample = X_test[0]
        pred = predict_sample(model, sample, target_names=target_names)
        print("Sample:", sample)
        print("Predicted class:", pred)


if __name__ == "__main__":
    main()
