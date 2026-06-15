# ML-Blog

Companion code for the machine learning blog series on aduwillie.com.

The repository is organized as standalone Python modules that demonstrate core ML topics using synthetic datasets and scikit-learn pipelines (with an optional TensorFlow/Keras section in Module 11).

## What Is In This Repo

- End-to-end examples for regression, classification, clustering, preprocessing, model evaluation, and feature engineering.
- Reproducible synthetic datasets generated directly in each module.
- Script-style modules you can run independently.

## Project Structure

- `module_01_foundations.py`: Baselines, regression, classification, and bias-variance tradeoff.
- `module_02_classification.py`: KNN, logistic regression, Gaussian Naive Bayes, and LDA comparison.
- `module_03_regression.py`: Linear, Ridge, Lasso, Elastic Net, and KNN regression comparison.
- `module_04_evaluation_validation.py`: Metrics, cross-validation, and GridSearchCV for tuning.
- `module_05_preprocessing.py`: Imputation, scaling, log transform, and encoding in a pipeline.
- `module_06_neural_networks.py`: MLPClassifier and MLPRegressor examples with scikit-learn.
- `module_07_training_neural_networks.py`: Neural-network training configuration comparisons.
- `module_08_clustering.py`: K-means, hierarchical clustering, and DBSCAN with silhouette scoring.
- `module_10_feature_selection_extraction.py`: SelectKBest, RFE, PCA, and KernelPCA.
- `module_11_keras_bridge.py`: TF-IDF + logistic regression baseline and optional Keras text model.
- `requirements.txt`: Base dependencies (TensorFlow is optional and only needed for Module 11 Keras section).

## Requirements

- Python 3.10+
- `pip`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

Run any module directly:

```bash
python module_01_foundations.py
python module_02_classification.py
python module_03_regression.py
python module_04_evaluation_validation.py
python module_05_preprocessing.py
python module_06_neural_networks.py
python module_07_training_neural_networks.py
python module_08_clustering.py
python module_10_feature_selection_extraction.py
python module_11_keras_bridge.py
```

Some modules support writing generated sample datasets to CSV:

```bash
python module_01_foundations.py --write-data
python module_02_classification.py --write-data
python module_03_regression.py --write-data
python module_05_preprocessing.py --write-data
python module_08_clustering.py --write-data
```

## Notes

- Module numbers follow the blog sequence; `module_09` is not present in this repository.
- Module 11 runs without TensorFlow by default (scikit-learn baseline still executes).
- To enable the Keras section in Module 11, install TensorFlow:

```bash
pip install tensorflow
```
