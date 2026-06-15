"""Module 6 companion script: neural networks with scikit-learn."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_moons, make_regression
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def classification_demo() -> None:
    X_array, y = make_moons(n_samples=900, noise=0.24, random_state=RANDOM_STATE)
    X = pd.DataFrame(X_array, columns=["signal_x", "signal_y"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(32, 16),
                    activation="relu",
                    alpha=0.001,
                    max_iter=800,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nMLP classification")
    print("------------------")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(classification_report(y_test, predictions))


def regression_demo() -> None:
    X_array, y = make_regression(
        n_samples=900,
        n_features=8,
        n_informative=6,
        noise=22.0,
        random_state=RANDOM_STATE,
    )
    y = y + 12 * np.sin(X_array[:, 0]) + 8 * np.maximum(X_array[:, 1], 0)
    X = pd.DataFrame(X_array, columns=[f"feature_{i}" for i in range(X_array.shape[1])])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    alpha=0.001,
                    max_iter=900,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nMLP regression")
    print("--------------")
    print(f"MAE: {mean_absolute_error(y_test, predictions):.2f}")
    print(f"R2 : {r2_score(y_test, predictions):.3f}")


def main() -> None:
    classification_demo()
    regression_demo()


if __name__ == "__main__":
    main()

