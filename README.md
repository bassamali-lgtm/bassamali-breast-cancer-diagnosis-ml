# Predicting Breast Cancer Diagnosis and Discovering Patient Subgroups

**Machine Learning and Data Mining for Breast Cancer Diagnosis**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Author

**Bassam Talib Sabri**
University of Information Technology and Communications
College of Business Informatics
Baghdad, Iraq

Corresponding author: [bassam.ali@uoitc.edu.iq](mailto:bassam.ali@uoitc.edu.iq)

---

## Overview

This repository contains the reproducibility materials and Python implementation for the research project:

> **Predicting Breast Cancer Diagnosis and Discovering Patient Subgroups**

The project applies supervised and unsupervised machine-learning techniques to the **Breast Cancer Wisconsin (Diagnostic)** dataset.

The supervised-learning component evaluates four classification algorithms:

1. Logistic Regression
2. Support Vector Machine with an RBF kernel
3. Random Forest
4. Gradient Boosting

The unsupervised-learning component applies:

1. Principal Component Analysis (PCA)
2. K-means clustering

The study evaluates classification performance using ROC-AUC, accuracy, precision, recall, and F1 score. The experimental protocol uses stratified 5-fold cross-validation and an 80/20 stratified holdout test. The analysis uses a fixed random seed of `42` for reproducibility.

---

## Dataset

The project uses the **Breast Cancer Wisconsin (Diagnostic)** dataset.

### Dataset characteristics

| Property           |                                  Value |
| ------------------ | -------------------------------------: |
| Number of cases    |                                    569 |
| Number of features |                                     30 |
| Benign cases       |                                    357 |
| Malignant cases    |                                    212 |
| Feature type       | Continuous numerical tumor descriptors |

The target coding used in the project is:

```text
0 = malignant
1 = benign
```

The dataset is available through scikit-learn's `load_breast_cancer()` dataset loader.

The repository does not require manually downloading a separate CSV file. The dataset can be generated locally using:

```bash
python src/prepare_data.py
```

---

## Methodology

### 1. Data preprocessing

The analysis uses standardization through Z-score normalization:

```text
mean = 0
standard deviation = 1
```

Standardization is particularly important for scale-sensitive algorithms such as Logistic Regression and RBF-SVM.

The train/test split uses stratified sampling to preserve the class distribution.

---

### 2. Supervised learning

Four machine-learning models are evaluated:

#### Logistic Regression

A linear classification model used as a baseline.

#### Support Vector Machine (RBF)

An SVM using the radial basis function kernel.

```text
kernel = RBF
probability = True
```

#### Random Forest

An ensemble of decision trees.

```text
n_estimators = 100
random_state = 42
```

#### Gradient Boosting

A gradient-boosting classifier using the default scikit-learn configuration with a fixed random seed.

---

## Evaluation

### Stratified 5-Fold Cross-Validation

The classification models are evaluated using:

```text
StratifiedKFold
n_splits = 5
shuffle = True
random_state = 42
```

The following metrics are reported:

* Accuracy
* Precision
* Recall
* F1 score
* ROC-AUC

Results are saved to:

```text
results/cross_validation_results.csv
```

---

## Holdout Test

After cross-validation, the selected model is evaluated using an independent stratified 20% holdout test set.

The split is:

```text
80% Training
20% Testing
```

with:

```text
random_state = 42
```

The holdout results are saved to:

```text
results/holdout_results.csv
```

The program also generates a confusion matrix and classification report.

---

## Unsupervised Learning

To investigate the structure of the dataset, PCA is applied before K-means clustering.

### PCA

The analysis uses:

```text
Number of components = 5
```

The research reports that five principal components retain approximately:

```text
84.7% of the total variance
```

### K-means

K-means clustering is then performed with:

```text
k = 2
random_state = 42
```

The clustering performance is evaluated using the **Adjusted Rand Index (ARI)**.

The research reports:

```text
ARI = 0.671
```

The clustering results are saved to:

```text
results/pca_kmeans_results.csv
```

---

## Reported Results

The accompanying research paper reports the following cross-validation results:

| Model               |        Accuracy |       Precision |          Recall |              F1 |         ROC-AUC |
| ------------------- | --------------: | --------------: | --------------: | --------------: | --------------: |
| Logistic Regression | 0.9719 ± 0.0129 | 0.9754 ± 0.0197 | 0.9804 ± 0.0143 | 0.9777 ± 0.0102 | 0.9951 ± 0.0055 |
| SVM (RBF)           | 0.9737 ± 0.0124 | 0.9834 ± 0.0161 | 0.9748 ± 0.0185 | 0.9789 ± 0.0099 | 0.9955 ± 0.0048 |
| Random Forest       | 0.9596 ± 0.0070 | 0.9650 ± 0.0245 | 0.9720 ± 0.0249 | 0.9679 ± 0.0055 | 0.9894 ± 0.0079 |
| Gradient Boosting   | 0.9456 ± 0.0217 | 0.9443 ± 0.0340 | 0.9720 ± 0.0152 | 0.9575 ± 0.0162 | 0.9920 ± 0.0051 |

The paper reports the RBF-SVM cross-validation ROC-AUC as:

```text
0.9955 ± 0.0048
```

The holdout results reported in the paper are:

| Metric    |  Value |
| --------- | -----: |
| Accuracy  | 0.9649 |
| Precision | 0.9857 |
| Recall    | 0.9583 |
| F1        | 0.9718 |
| ROC-AUC   | 0.9940 |

The paper reports that the selected RBF-SVM produced one malignant case incorrectly classified as benign in the holdout test.

---

## Reproducibility

### Requirements

Install Python 3.9 or later.

Install the required packages:

```bash
pip install -r requirements.txt
```

The main dependencies are:

```text
numpy
pandas
scikit-learn
```

---

## Running the Project

### Step 1 — Prepare the dataset

Run:

```bash
python src/prepare_data.py
```

This creates:

```text
data/breast_cancer_wisconsin_diagnostic.csv
```

---

### Step 2 — Run the analysis

Run:

```bash
python src/train_and_evaluate.py
```

The program performs:

```text
1. Dataset loading
2. Data preprocessing
3. Stratified 5-fold cross-validation
4. Model comparison
5. Best-model selection
6. 80/20 holdout evaluation
7. PCA
8. K-means clustering
9. Adjusted Rand Index calculation
10. Saving results
```

---

## Output Files

After execution, the following files are generated:

```text
results/
│
├── cross_validation_results.csv
├── holdout_results.csv
└── pca_kmeans_results.csv
```

### Cross-validation results

```text
results/cross_validation_results.csv
```

Contains:

* Accuracy
* Precision
* Recall
* F1
* ROC-AUC
* Standard deviations

### Holdout results

```text
results/holdout_results.csv
```

Contains:

* Accuracy
* Precision
* Recall
* F1
* ROC-AUC

### PCA and K-means results

```text
results/pca_kmeans_results.csv
```

Contains:

* Number of PCA components
* Explained variance
* Number of clusters
* Adjusted Rand Index

---

## Project Structure

```text
bassamali-breast-cancer-diagnosis-ml/
│
├── README.md
├── DATA_AVAILABILITY.md
├── CITATION.cff
├── CITATION.bib
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── data/
│   └── README.md
│
├── results/
│   └── README.md
│
├── notebooks/
│   └── README.md
│
└── src/
    ├── prepare_data.py
    └── train_and_evaluate.py
```

---

## Limitations

The study has several limitations.

First, the analysis uses a single benchmark dataset. Therefore, the reported results should not be interpreted as evidence of clinical performance on other institutions, populations, imaging systems, or clinical environments.

Second, the study does not include external validation.

Third, the PCA and K-means analysis is exploratory. The resulting clusters should not be interpreted as clinically validated patient subgroups without further investigation and domain expertise.

These limitations are consistent with those described in the accompanying research paper.

---

## Future Work

Potential future work includes:

* External validation using independent datasets.
* Systematic hyperparameter optimization.
* Evaluation on data from different institutions.
* Additional explainability methods.
* Further investigation of patient subgroups.
* Validation of clustering results with domain expertise.
* Comparison with additional machine-learning methods.

---

## Data Availability

The source code and reproducibility materials are publicly available through this GitHub repository:

```text
https://github.com/bassamali-lgtm/bassamali-breast-cancer-diagnosis-ml
```

The dataset is accessed through scikit-learn's dataset loader and is not redistributed as a separate dataset file in this repository.

The repository will be archived through Zenodo after the first public release.

**Zenodo DOI: To be added after publication of the repository release.**

---

## Citation

If you use this repository or its code, please cite:

```text
Sabri, Bassam Talib. Predicting Breast Cancer Diagnosis and
Discovering Patient Subgroups. GitHub repository, 2026.
```

A version-specific DOI will be added after the repository is archived and published through Zenodo.

See:

```text
CITATION.cff
CITATION.bib
```

GitHub supports citation files such as `CITATION.cff` to help users cite software repositories correctly.

---

## License

This project is distributed under the **MIT License**.

See the [LICENSE](LICENSE) file for details.

---

## Repository

GitHub:

https://github.com/bassamali-lgtm/bassamali-breast-cancer-diagnosis-ml

---

## Contact

**Bassam Talib Sabri**

University of Information Technology and Communications
College of Business Informatics
Baghdad, Iraq

Email:

```text
bassam.ali@uoitc.edu.iq
```
