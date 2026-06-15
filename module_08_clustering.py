"""Module 8 companion script: clustering with scikit-learn."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def make_player_data(random_state: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    archetypes = {
        "scoring_guard": [26, 5, 4, 2, 8, 32],
        "floor_general": [16, 10, 4, 2, 5, 31],
        "three_and_d_wing": [15, 3, 6, 3, 7, 29],
        "interior_anchor": [14, 2, 11, 5, 1, 28],
    }
    rows = []
    for name, center in archetypes.items():
        points = rng.normal(center, [4, 2, 2, 1.2, 1.5, 4], size=(90, len(center)))
        for row in points:
            rows.append(
                {
                    "points_per_game": max(row[0], 0),
                    "assists_per_game": max(row[1], 0),
                    "rebounds_per_game": max(row[2], 0),
                    "defensive_activity": max(row[3], 0),
                    "three_point_attempts": max(row[4], 0),
                    "minutes_per_game": max(row[5], 5),
                    "simulated_archetype": name,
                }
            )
    return pd.DataFrame(rows).round(2)


def describe_clusters(df: pd.DataFrame, labels: np.ndarray, title: str) -> None:
    result = df.drop(columns=["simulated_archetype"]).copy()
    result["cluster"] = labels
    print(f"\n{title}")
    print("-" * len(title))
    print(result.groupby("cluster").mean().round(2))
    valid_labels = labels[labels >= 0]
    if len(set(valid_labels)) > 1:
        features = df.drop(columns=["simulated_archetype"])
        scaled = StandardScaler().fit_transform(features)
        print(f"Silhouette score: {silhouette_score(scaled[labels >= 0], valid_labels):.3f}")
    if (labels == -1).any():
        print(f"Noise points: {(labels == -1).sum()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare clustering algorithms.")
    parser.add_argument("--write-data", action="store_true", help="Write the generated dataset to CSV.")
    args = parser.parse_args()

    df = make_player_data()
    if args.write_data:
        output = Path(__file__).with_name("module_08_player_profiles.csv")
        df.to_csv(output, index=False)
        print(f"Wrote {output}")

    X = df.drop(columns=["simulated_archetype"])
    scaled = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init="auto")
    describe_clusters(df, kmeans.fit_predict(scaled), "K-means clusters")

    hierarchical = AgglomerativeClustering(n_clusters=4)
    describe_clusters(df, hierarchical.fit_predict(scaled), "Hierarchical clusters")

    dbscan = DBSCAN(eps=0.95, min_samples=8)
    describe_clusters(df, dbscan.fit_predict(scaled), "DBSCAN clusters")


if __name__ == "__main__":
    main()

