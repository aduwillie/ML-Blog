"""Module 5 companion script: preprocessing and feature engineering."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


RANDOM_STATE = 42


def make_readmission_data(n_rows: int = 1000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    age = rng.normal(61, 16, n_rows).clip(18, 95)
    previous_visits = rng.poisson(2.2, n_rows)
    medication_count = rng.poisson(5.0, n_rows).clip(0, 24)
    lab_risk_score = rng.normal(0, 1, n_rows)
    length_of_stay = rng.gamma(2.0, 2.5, n_rows).clip(1, 30)
    followup_scheduled = rng.choice([0, 1], size=n_rows, p=[0.35, 0.65])
    discharge_type = rng.choice(["home", "home_health", "skilled_nursing"], size=n_rows, p=[0.68, 0.21, 0.11])
    insurance_type = rng.choice(["private", "medicare", "medicaid", "self_pay"], size=n_rows)

    logit = (
        -2.2
        + 0.018 * age
        + 0.22 * previous_visits
        + 0.11 * medication_count
        + 0.75 * lab_risk_score
        + 0.08 * length_of_stay
        - 0.55 * followup_scheduled
        + np.where(discharge_type == "skilled_nursing", 0.45, 0)
        + np.where(insurance_type == "self_pay", 0.35, 0)
        + rng.normal(0, 0.35, n_rows)
    )
    readmitted = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    df = pd.DataFrame(
        {
            "age": age.round(0),
            "previous_visits": previous_visits,
            "medication_count": medication_count,
            "lab_risk_score": lab_risk_score.round(2),
            "length_of_stay": length_of_stay.round(1),
            "followup_scheduled": followup_scheduled,
            "discharge_type": discharge_type,
            "insurance_type": insurance_type,
            "readmitted": readmitted,
        }
    )

    for column in ["age", "lab_risk_score", "discharge_type"]:
        missing = rng.choice(df.index, size=int(0.06 * n_rows), replace=False)
        df.loc[missing, column] = np.nan
    return df


def build_pipeline() -> Pipeline:
    numeric_features = ["age", "lab_risk_score", "followup_scheduled"]
    skewed_features = ["previous_visits", "medication_count", "length_of_stay"]
    categorical_features = ["discharge_type", "insurance_type"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    skewed_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("skew", skewed_pipeline, skewed_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate preprocessing pipelines.")
    parser.add_argument("--write-data", action="store_true", help="Write the generated dataset to CSV.")
    args = parser.parse_args()

    df = make_readmission_data()
    if args.write_data:
        output = Path(__file__).with_name("module_05_readmission.csv")
        df.to_csv(output, index=False)
        print(f"Wrote {output}")

    X = df.drop(columns=["readmitted"])
    y = df["readmitted"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    model = build_pipeline()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("Readmission preprocessing pipeline")
    print("----------------------------------")
    print(f"ROC AUC: {roc_auc_score(y_test, probabilities):.3f}")
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    main()

