import os
from pathlib import Path
import sys

# Ensure project root is importable when pytest runs from different CWDs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from iris_classifier import train_model, load_data, save_model, load_model, evaluate_model


def test_train_save_load(tmp_path: Path):
    X, y, _ = load_data()
    # quick small model
    model = train_model(X, y, n_estimators=10)
    assert hasattr(model, "predict")

    out = tmp_path / "model.joblib"
    save_model(model, out)
    assert out.exists()

    loaded = load_model(out)
    # ensure loaded model predicts same as original
    x = X[0]
    assert int(loaded.predict(x.reshape(1, -1))[0]) == int(model.predict(x.reshape(1, -1))[0])


def test_evaluate_accuracy():
    X, y, target_names = load_data()
    # train/test split same as module
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = train_model(X_train, y_train, n_estimators=50)
    res = evaluate_model(model, X_test, y_test, target_names=target_names)
    assert res["accuracy"] >= 0.8
