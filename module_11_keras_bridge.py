"""Module 11 companion script: scikit-learn baseline and optional Keras text model."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42


def make_review_data(repeats: int = 80) -> tuple[np.ndarray, np.ndarray]:
    positive_templates = [
        "great battery life and easy setup",
        "excellent quality and fast delivery",
        "comfortable design with reliable performance",
        "the product works beautifully and feels premium",
        "clear instructions and very useful features",
        "durable build and wonderful customer experience",
    ]
    negative_templates = [
        "poor battery life and confusing setup",
        "bad quality and slow delivery",
        "uncomfortable design with unreliable performance",
        "the product broke quickly and feels cheap",
        "unclear instructions and missing features",
        "fragile build and frustrating customer experience",
    ]

    texts: list[str] = []
    labels: list[int] = []
    for i in range(repeats):
        for text in positive_templates:
            texts.append(f"{text} order {i}")
            labels.append(1)
        for text in negative_templates:
            texts.append(f"{text} order {i}")
            labels.append(0)
    return np.array(texts), np.array(labels)


def sklearn_baseline(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> None:
    baseline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    baseline.fit(X_train, y_train)
    predictions = baseline.predict(X_test)

    print("scikit-learn TF-IDF baseline")
    print("----------------------------")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(classification_report(y_test, predictions, target_names=["negative", "positive"]))


def keras_model(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> None:
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        print("\nKeras portion skipped.")
        print("Install TensorFlow to run it: pip install tensorflow")
        return

    tf.random.set_seed(RANDOM_STATE)

    vectorizer = layers.TextVectorization(max_tokens=5000, output_sequence_length=30)
    vectorizer.adapt(X_train)

    model = keras.Sequential(
        [
            vectorizer,
            layers.Embedding(input_dim=5000, output_dim=32),
            layers.GlobalAveragePooling1D(),
            layers.Dense(16, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        )
    ]

    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=12,
        batch_size=32,
        callbacks=callbacks,
        verbose=0,
    )

    probabilities = model.predict(X_test, verbose=0).reshape(-1)
    predictions = (probabilities >= 0.5).astype(int)

    print("\nKeras text model")
    print("----------------")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(classification_report(y_test, predictions, target_names=["negative", "positive"]))


def main() -> None:
    X, y = make_review_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    sklearn_baseline(X_train, X_test, y_train, y_test)
    keras_model(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()

