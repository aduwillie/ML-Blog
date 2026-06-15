"""Module 10 companion script: feature selection and extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA, PCA
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def make_nutrition_data(n_rows: int = 700, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    calories = rng.normal(2200, 450, n_rows).clip(1000, 4200)
    protein = calories / 90 + rng.normal(0, 8, n_rows)
    carbs = calories / 8 + rng.normal(0, 35, n_rows)
    fat = calories / 30 + rng.normal(0, 10, n_rows)
    fiber = rng.normal(24, 8, n_rows).clip(4, 60)
    sodium = rng.normal(2600, 700, n_rows).clip(700, 6000)
    vitamin_c = rng.gamma(2.0, 35, n_rows)
    calcium = rng.normal(900, 220, n_rows).clip(200, 1800)
    added_sugar = rng.gamma(2.0, 18, n_rows)
    exercise_minutes = rng.gamma(2.0, 20, n_rows)
    sleep_hours = rng.normal(7, 1.1, n_rows).clip(3, 10)
    stress = rng.integers(1, 11, n_rows)

    health_score = (
        45
        + 0.16 * fiber
        + 0.018 * vitamin_c
        + 0.004 * calcium
        - 0.006 * sodium
        - 0.11 * added_sugar
        + 0.06 * exercise_minutes
        + 2.2 * sleep_hours
        - 1.7 * stress
        + rng.normal(0, 5.0, n_rows)
    )

    return pd.DataFrame(
        {
            "calories": calories,
            "protein": protein,
            "carbs": carbs,
            "fat": fat,
            "fiber": fiber,
            "sodium": sodium,
            "vitamin_c": vitamin_c,
            "calcium": calcium,
            "added_sugar": added_sugar,
            "exercise_minutes": exercise_minutes,
            "sleep_hours": sleep_hours,
            "stress": stress,
            "health_score": health_score,
        }
    ).round(2)


def evaluate(name: str, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> None:
    scores = cross_val_score(pipeline, X, y, cv=5, scoring="neg_mean_absolute_error")
    print(f"{name:32s} mean CV MAE={-scores.mean():.2f}")


def main() -> None:
    df = make_nutrition_data()
    X = df.drop(columns=["health_score"])
    y = df["health_score"]

    pipelines = {
        "all features": Pipeline([("scaler", StandardScaler()), ("regressor", Ridge())]),
        "select k best": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(score_func=f_regression, k=6)),
                ("regressor", Ridge()),
            ]
        ),
        "recursive feature elimination": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("selector", RFE(estimator=Ridge(), n_features_to_select=6)),
                ("regressor", Ridge()),
            ]
        ),
        "pca extraction": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=5)),
                ("regressor", Ridge()),
            ]
        ),
        "kernel pca extraction": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("kpca", KernelPCA(n_components=5, kernel="rbf", gamma=0.05)),
                ("regressor", Ridge()),
            ]
        ),
    }

    print("Feature processing comparison")
    print("-----------------------------")
    for name, pipeline in pipelines.items():
        evaluate(name, pipeline, X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )
    selector_pipeline = pipelines["select k best"]
    selector_pipeline.fit(X_train, y_train)
    predictions = selector_pipeline.predict(X_test)
    print(f"\nHeld-out MAE for SelectKBest pipeline: {mean_absolute_error(y_test, predictions):.2f}")


if __name__ == "__main__":
    main()

