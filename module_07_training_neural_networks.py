"""Module 7 companion script: neural-network training options."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def main() -> None:
    X_array, y = make_classification(
        n_samples=1500,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        class_sep=0.9,
        random_state=RANDOM_STATE,
    )
    X = pd.DataFrame(X_array, columns=[f"feature_{i}" for i in range(X_array.shape[1])])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    configs = {
        "small network": dict(hidden_layer_sizes=(16,), alpha=0.0001, learning_rate_init=0.001),
        "larger network": dict(hidden_layer_sizes=(64, 32), alpha=0.0001, learning_rate_init=0.001),
        "stronger regularization": dict(hidden_layer_sizes=(64, 32), alpha=0.01, learning_rate_init=0.001),
        "early stopping": dict(
            hidden_layer_sizes=(64, 32),
            alpha=0.001,
            learning_rate_init=0.001,
            early_stopping=True,
            validation_fraction=0.15,
        ),
        "larger learning rate": dict(hidden_layer_sizes=(64, 32), alpha=0.001, learning_rate_init=0.01),
    }

    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    print("Neural-network training comparison")
    print("----------------------------------")
    for name, kwargs in configs.items():
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        max_iter=500,
                        random_state=RANDOM_STATE,
                        solver="adam",
                        **kwargs,
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print(f"{name:25s} accuracy={accuracy_score(y_test, predictions):.3f} f1={f1_score(y_test, predictions):.3f}")


if __name__ == "__main__":
    main()

