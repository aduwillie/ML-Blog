"""Module 4 companion script: metrics, cross-validation, and tuning."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification, make_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def numeric_preprocessor(columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                columns,
            )
        ]
    )


def classification_demo() -> None:
    X_array, y = make_classification(
        n_samples=1200,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        weights=[0.72, 0.28],
        class_sep=1.0,
        random_state=RANDOM_STATE,
    )
    columns = [f"signal_{i}" for i in range(X_array.shape[1])]
    X = pd.DataFrame(X_array, columns=columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    model = Pipeline(
        steps=[
            ("preprocess", numeric_preprocessor(columns)),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nClassification evaluation")
    print("-------------------------")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(f"ROC AUC : {roc_auc_score(y_test, probabilities):.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))

    search = GridSearchCV(
        estimator=model,
        param_grid={"classifier__C": [0.05, 0.1, 1.0, 10.0]},
        cv=5,
        scoring="f1",
    )
    search.fit(X_train, y_train)
    print("Best logistic regression C:", search.best_params_["classifier__C"])
    print(f"Best cross-validated F1 : {search.best_score_:.3f}")


def regression_demo() -> None:
    X_array, y = make_regression(
        n_samples=900,
        n_features=7,
        n_informative=5,
        noise=18.0,
        random_state=RANDOM_STATE,
    )
    columns = [f"feature_{i}" for i in range(X_array.shape[1])]
    X = pd.DataFrame(X_array, columns=columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )

    model = Pipeline(
        steps=[
            ("preprocess", numeric_preprocessor(columns)),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nRegression evaluation")
    print("---------------------")
    print(f"MAE : {mean_absolute_error(y_test, predictions):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):.2f}")
    print(f"R2  : {r2_score(y_test, predictions):.3f}")

    scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="neg_mean_absolute_error",
    )
    print(f"Cross-validated MAE by fold: {np.round(-scores, 2)}")
    print(f"Mean cross-validated MAE   : {-scores.mean():.2f}")


def main() -> None:
    classification_demo()
    regression_demo()


if __name__ == "__main__":
    main()

