"""Module 2 companion script: classification models with scikit-learn."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42


def make_passenger_satisfaction_data(n_rows: int = 900, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    travel_type = rng.choice(["business", "personal"], size=n_rows, p=[0.58, 0.42])
    customer_type = rng.choice(["loyal", "first_time"], size=n_rows, p=[0.72, 0.28])
    flight_distance = rng.gamma(shape=2.2, scale=550, size=n_rows).clip(100, 4800)
    delay_minutes = rng.gamma(shape=1.3, scale=18, size=n_rows).clip(0, 180)
    seat_comfort = rng.integers(1, 6, size=n_rows)
    inflight_service = rng.integers(1, 6, size=n_rows)
    online_boarding = rng.integers(1, 6, size=n_rows)

    score = (
        -1.6
        + 0.55 * seat_comfort
        + 0.48 * inflight_service
        + 0.42 * online_boarding
        - 0.025 * delay_minutes
        + 0.00025 * flight_distance
        + np.where(travel_type == "business", 0.45, -0.15)
        + np.where(customer_type == "loyal", 0.35, -0.25)
        + rng.normal(0, 0.9, n_rows)
    )
    probability = 1 / (1 + np.exp(-score))
    satisfied = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "travel_type": travel_type,
            "customer_type": customer_type,
            "flight_distance": flight_distance.round(0),
            "delay_minutes": delay_minutes.round(0),
            "seat_comfort": seat_comfort,
            "inflight_service": inflight_service,
            "online_boarding": online_boarding,
            "satisfied": satisfied,
        }
    )


def build_preprocessor() -> ColumnTransformer:
    numeric_features = [
        "flight_distance",
        "delay_minutes",
        "seat_comfort",
        "inflight_service",
        "online_boarding",
    ]
    categorical_features = ["travel_type", "customer_type"]

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare common classification models.")
    parser.add_argument("--write-data", action="store_true", help="Write the generated dataset to CSV.")
    args = parser.parse_args()

    df = make_passenger_satisfaction_data()
    features = [
        "travel_type",
        "customer_type",
        "flight_distance",
        "delay_minutes",
        "seat_comfort",
        "inflight_service",
        "online_boarding",
    ]
    X = df[features]
    y = df["satisfied"]

    if args.write_data:
        output = Path(__file__).with_name("module_02_passenger_satisfaction.csv")
        df.to_csv(output, index=False)
        print(f"Wrote {output}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "k-nearest neighbors": KNeighborsClassifier(n_neighbors=9),
        "logistic regression": LogisticRegression(max_iter=1000),
        "gaussian naive bayes": GaussianNB(),
        "linear discriminant analysis": LinearDiscriminantAnalysis(),
    }

    for name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        print(f"\n{name.title()}")
        print("-" * len(name))
        print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
        print("Confusion matrix:")
        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions, target_names=["not satisfied", "satisfied"]))


if __name__ == "__main__":
    main()

