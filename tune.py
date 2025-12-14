"""Hyperparameter tuning for the Iris RandomForest classifier.

Saves the best model to models/iris_rf_tuned.joblib
"""
from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def main(out_path: Path = Path("models/iris_rf_tuned.joblib")):
    iris = load_iris()
    X, y = iris.data, iris.target

    param_dist = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [None, 3, 5, 8, 12],
        "min_samples_split": [2, 3, 4, 5],
        "min_samples_leaf": [1, 2, 3],
        "bootstrap": [True, False],
    }

    rf = RandomForestClassifier(random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=20, cv=cv, n_jobs=-1, random_state=42)
    search.fit(X, y)

    print("Best params:", search.best_params_)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, out_path)
    print(f"Saved tuned model to {out_path}")


if __name__ == "__main__":
    main()
