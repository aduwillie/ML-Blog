"""Module 3 companion script: regression models with scikit-learn."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42


def make_wellbeing_data(n_rows: int = 750, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    sleep_hours = rng.normal(7.0, 1.2, n_rows).clip(3.5, 10.5)
    exercise_minutes = rng.gamma(2.0, 18.0, n_rows).clip(0, 140)
    screen_hours = rng.normal(5.2, 1.8, n_rows).clip(0.5, 12)
    stress_level = rng.integers(1, 11, n_rows)
    social_support = rng.integers(1, 11, n_rows)
    work_mode = rng.choice(["onsite", "remote", "hybrid"], size=n_rows, p=[0.42, 0.28, 0.30])

    wellbeing = (
        52
        + 3.2 * sleep_hours
        + 0.10 * exercise_minutes
        - 1.3 * screen_hours
        - 2.4 * stress_level
        + 1.8 * social_support
        + np.where(work_mode == "hybrid", 2.0, 0)
        + np.where(work_mode == "remote", 0.8, 0)
        + rng.normal(0, 5.0, n_rows)
    ).clip(0, 100)

    return pd.DataFrame(
        {
            "sleep_hours": sleep_hours.round(1),
            "exercise_minutes": exercise_minutes.round(0),
            "screen_hours": screen_hours.round(1),
            "stress_level": stress_level,
            "social_support": social_support,
            "work_mode": work_mode,
            "wellbeing_score": wellbeing.round(1),
        }
    )


def build_preprocessor() -> ColumnTransformer:
    numeric = ["sleep_hours", "exercise_minutes", "screen_hours", "stress_level", "social_support"]
    categorical = ["work_mode"]
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
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )


def report(name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    predictions = model.predict(X_test)
    print(f"\n{name}")
    print("-" * len(name))
    print(f"MAE : {mean_absolute_error(y_test, predictions):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):.2f}")
    print(f"R2  : {r2_score(y_test, predictions):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare common regression models.")
    parser.add_argument("--write-data", action="store_true", help="Write the generated dataset to CSV.")
    args = parser.parse_args()

    df = make_wellbeing_data()
    if args.write_data:
        output = Path(__file__).with_name("module_03_wellbeing.csv")
        df.to_csv(output, index=False)
        print(f"Wrote {output}")

    features = [
        "sleep_hours",
        "exercise_minutes",
        "screen_hours",
        "stress_level",
        "social_support",
        "work_mode",
    ]
    X = df[features]
    y = df["wellbeing_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )

    models = {
        "linear regression": LinearRegression(),
        "ridge regression": Ridge(alpha=1.0),
        "lasso regression": Lasso(alpha=0.05),
        "elastic net": ElasticNet(alpha=0.05, l1_ratio=0.5),
        "k-nearest neighbors regression": KNeighborsRegressor(n_neighbors=9),
    }

    for name, estimator in models.items():
        pipeline = Pipeline(steps=[("preprocess", build_preprocessor()), ("regressor", estimator)])
        pipeline.fit(X_train, y_train)
        report(name.title(), pipeline, X_test, y_test)


if __name__ == "__main__":
    main()

