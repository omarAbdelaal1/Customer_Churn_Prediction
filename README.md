# Customer Churn Prediction

**Author:** Omar Abdelaal

Predict whether a bank customer will stay or leave using demographic and account information. This project includes data preprocessing, exploratory data analysis (EDA), and modeling with XGBoost, Random Forest, and Extra Trees, using hyperparameter tuning and threshold optimization for improved performance.

---

## Table of Contents

* Project Overview
* Dataset
* Features
* Data Preprocessing
* Exploratory Data Analysis
* Modeling
* Evaluation
* Installation
* Usage
* Results
* Saving the Model

---

## Project Overview

Customer Churn Prediction aims to identify customers likely to leave a bank, helping businesses implement strategies to retain valuable customers. The target variable is `Exited`, making this a binary classification task.

---

## Dataset

The dataset contains 10,000 rows and 14 columns, capturing customer demographics, account details, and banking behavior.

---

## Features

* `RowNumber`, `CustomerId`, `Surname` – Identifiers (dropped before modeling)
* `CreditScore` – Customer's credit score
* `Geography` – Country
* `Gender` – Male/Female
* `Age` – Customer age
* `Tenure` – Years with the bank
* `Balance` – Account balance
* `NumOfProducts` – Number of products owned
* `HasCrCard` – Credit card ownership
* `IsActiveMember` – Activity status
* `EstimatedSalary` – Estimated annual salary
* `Exited` – Target: 0 = stayed, 1 = left

---

## Data Preprocessing

* Label encoding for categorical features (`Gender`, `Geography`)
* Dropped non-informative identifiers
* Outlier removal using Z-score
* Resampling with Tomek Links to handle class imbalance
* Standardization of numeric features

---

## Exploratory Data Analysis (EDA)

* Univariate, bivariate, and multivariate analysis
* Visualizations include count plots, histograms, scatter plots, correlation heatmaps, and pie charts
* Observed key trends:

  * Customers with 3+ products or from certain geographies are more likely to churn
  * Balance, Age, and Tenure influence churn probability

---

## Modeling

Three models are implemented:

1. **XGBoost** – Hyperparameter tuning with Optuna, threshold optimization for F1 Score
2. **Random Forest** – Optuna tuning for ROC-AUC, balanced class weights
3. **Extra Trees** – Optuna tuning for ROC-AUC, balanced class weights

---

## Evaluation

Models evaluated using:

* Accuracy
* Balanced Accuracy
* F1 Score
* ROC-AUC
* Confusion Matrix

Comparison table generated for all three models.

---

## Usage

```python
import pickle

# Load saved XGBoost model
with open('churn.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict on new data
y_pred = model.predict(X_new)
```

---

## Results

* XGBoost achieved the best F1 score after threshold tuning
* Random Forest and Extra Trees also performed well on ROC-AUC
* Confusion matrices and evaluation metrics provided for model comparison

---

## Saving the Model

The final XGBoost model is saved as `churn.pkl` for deployment.
