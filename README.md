# Supervised and Unsupervised Machine Learning

## Objective
Build and evaluate classification, regression, and clustering models using a synthetic loan-default dataset.

## Dataset
- **File:** `loan_default.csv`
- **Target column:** `default`
- `0` — No default
- `1` — Default

The dataset contains synthetic data for machine learning practice.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Matplotlib

## 1. Classification Models

Four classification algorithms were trained and evaluated using validation accuracy.

| Model | Validation Accuracy |
|---|---:|
| Logistic Regression | 87.0% |
| Decision Tree | 88.0% |
| Random Forest | 87.5% |
| XGBoost | 92.0% |

**Best validation accuracy:** XGBoost — 92.0%

- Random Forest OOB score: `0.8667`
- XGBoost best iteration: `135`

## 2. Prediction Probabilities

Logistic Regression generated class predictions and probabilities for five validation rows.

Each row contains:
- Probability of class 0 (no default)
- Probability of class 1 (default)

## 3. Feature Importance

### Random Forest — Top 5 Features
1. Employment years
2. Debt-to-income
3. Late payments
4. Income-to-loan ratio
5. Income

### XGBoost — Top 5 Features
1. Late payments
2. Employment years
3. Income-to-loan ratio
4. Loan amount
5. Debt-to-income

Feature importance indicates how the trained models use the input features; it does not prove causation.

## 4. Regression Models

Linear Regression, Ridge, and Lasso were evaluated using validation R-squared.

| Model | Validation R-squared |
|---|---:|
| Linear Regression | 0.1485 |
| Ridge | 0.1485 |
| Lasso | 0.1485 |

- Lasso zero coefficients: `0`

The regression results are illustrative because the dataset and loan amounts are synthetic.

## 5. K-Means Clustering

K-Means was tested for cluster counts from 2 to 8 using income, loan amount, and credit score.

| Number of Clusters (k) | Inertia | Silhouette Score |
|---:|---:|---:|
| 2 | 2141.18 | 0.2819 |
| 3 | 1687.85 | 0.2556 |
| 4 | 1327.52 | 0.2803 |
| 5 | 1172.02 | 0.2616 |
| 6 | 1048.55 | 0.2522 |
| 7 | 948.40 | 0.2498 |
| 8 | 869.22 | 0.2536 |

The highest silhouette score was achieved at `k = 2`, with a score of `0.2819`. The score at `k = 4` was very similar, at `0.2803`.

These scores suggest that the clusters have limited separation.

### Generated Plots
- `kmeans_elbow.png`
- `kmeans_silhouette.png`

## 6. Output Files
- `task3.py`
- `loan_default.csv`
- `task3_classification_results.csv`
- `kmeans_elbow.png`
- `kmeans_silhouette.png`
- `README.md`

## 7. How to Run

Install the required libraries:

```bash
python -m pip install pandas numpy matplotlib scikit-learn xgboost
```

Run the program:

```bash
python task3.py
```

## Conclusion

Four classification models, three regression models, and K-Means clustering were evaluated. XGBoost achieved the highest validation accuracy at 92.0%. The clustering results suggest that the customer segments are not strongly separated in this synthetic dataset.
