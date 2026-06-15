"""Module 1 companion script: machine learning foundations with scikit-learn.

This script creates a synthetic coffee-demand dataset, trains regression and
classification models, compares against baselines, and demonstrates the
bias-variance tradeoff with polynomial features.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler


RANDOM_STATE = 42


def make_coffee_demand_data(n_days: int = 365, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Create a reproducible synthetic dataset for a coffee shop."""
    rng = np.random.default_rng(random_state)

    day_numbers = np.arange(n_days)
    day_names = np.array(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
    day_of_week = day_names[day_numbers % 7]

    seasonal_wave = np.sin(2 * np.pi * day_numbers / 365)
    temperature_f = 62 + 18 * seasonal_wave + rng.normal(0, 6, n_days)
    rain_inches = np.clip(rng.gamma(shape=1.2, scale=0.18, size=n_days) - 0.05, 0, None)
    event_score = rng.beta(a=1.2, b=4.0, size=n_days)
    is_holiday = rng.choice([0, 1], size=n_days, p=[0.94, 0.06])
    weekend = np.isin(day_of_week, ["Saturday", "Sunday"]).astype(int)

    base_orders = 95
    orders = (
        base_orders
        + 0.9 * temperature_f
        - 18 * rain_inches
        + 55 * event_score
        + 24 * weekend
        + 30 * is_holiday
        + rng.normal(0, 12, n_days)
    )
    orders = np.maximum(orders.round(), 20).astype(int)

    return pd.DataFrame(
        {
            "temperature_f": temperature_f.round(1),
            "rain_inches": rain_inches.round(2),
            "event_score": event_score.round(2),
            "is_holiday": is_holiday,
            "day_of_week": day_of_week,
            "orders": orders,
        }
    )


def build_preprocessor() -> ColumnTransformer:
    """Build reusable preprocessing for numeric and categorical features."""
    numeric_features = ["temperature_f", "rain_inches", "event_score", "is_holiday"]
    categorical_features = ["day_of_week"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def evaluate_regression_model(name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Print regression metrics in business-friendly units."""
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"MAE : {mae:6.2f} orders")
    print(f"RMSE: {rmse:6.2f} orders")
    print(f"R^2 : {r2:6.3f}")


def run_regression_workflow(df: pd.DataFrame) -> None:
    """Train a baseline and a Ridge regression model."""
    feature_columns = ["temperature_f", "rain_inches", "event_score", "is_holiday", "day_of_week"]
    X = df[feature_columns]
    y = df["orders"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    baseline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("regressor", DummyRegressor(strategy="mean")),
        ]
    )

    ridge_model = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )

    baseline.fit(X_train, y_train)
    ridge_model.fit(X_train, y_train)

    evaluate_regression_model("Baseline: always predict the average", baseline, X_test, y_test)
    evaluate_regression_model("Ridge regression pipeline", ridge_model, X_test, y_test)

    cv_scores = cross_val_score(
        ridge_model,
        X,
        y,
        scoring="neg_mean_absolute_error",
        cv=5,
    )
    print("\nRidge cross-validation")
    print("----------------------")
    print(f"MAE by fold: {np.round(-cv_scores, 2)}")
    print(f"Mean MAE   : {-cv_scores.mean():.2f} orders")


def run_classification_workflow(df: pd.DataFrame) -> None:
    """Convert demand prediction into a high-demand classification problem."""
    feature_columns = ["temperature_f", "rain_inches", "event_score", "is_holiday", "day_of_week"]
    X = df[feature_columns]
    y = (df["orders"] >= 175).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    baseline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("classifier", DummyClassifier(strategy="most_frequent")),
        ]
    )

    classifier = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    baseline.fit(X_train, y_train)
    classifier.fit(X_train, y_train)

    baseline_predictions = baseline.predict(X_test)
    classifier_predictions = classifier.predict(X_test)

    print("\nClassification: high-demand day")
    print("--------------------------------")
    print(f"Baseline accuracy          : {accuracy_score(y_test, baseline_predictions):.3f}")
    print(f"Logistic regression accuracy: {accuracy_score(y_test, classifier_predictions):.3f}")
    print("\nDetailed logistic regression report")
    print(classification_report(y_test, classifier_predictions, target_names=["normal", "high"]))


def demonstrate_bias_variance(df: pd.DataFrame) -> None:
    """Compare simple and complex polynomial models on one numeric feature."""
    X = df[["temperature_f"]]
    y = df["orders"]

    print("\nBias-variance demonstration")
    print("---------------------------")
    for degree in [1, 3, 12]:
        model = Pipeline(
            steps=[
                ("polynomial_features", PolynomialFeatures(degree=degree, include_bias=False)),
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=1.0)),
            ]
        )
        scores = cross_val_score(model, X, y, scoring="neg_mean_absolute_error", cv=5)
        print(f"Polynomial degree {degree:>2}: mean CV MAE = {-scores.mean():.2f} orders")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Module 1 scikit-learn workflow.")
    parser.add_argument(
        "--write-data",
        action="store_true",
        help="Write the generated coffee demand dataset to samples/module_01_coffee_demand.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = make_coffee_demand_data()

    print("Synthetic coffee-demand dataset")
    print("-------------------------------")
    print(df.head())
    print(f"\nRows: {len(df)}")

    if args.write_data:
        output_path = Path(__file__).with_name("module_01_coffee_demand.csv")
        df.to_csv(output_path, index=False)
        print(f"\nWrote sample dataset to: {output_path}")

    run_regression_workflow(df)
    run_classification_workflow(df)
    demonstrate_bias_variance(df)


if __name__ == "__main__":
    main()
