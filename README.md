# Bank Customer Churn Classification

Predicts whether a bank customer will churn (`Exited`), comparing a baseline decision tree against a CV-tuned tree, a Random Forest, and Logistic Regression on 10,000 customers.

**Read the [Key Caveat](#key-caveat-the-original-model-was-fake) section before reusing any number from this project.**

> **Companion project:** [`bank-churn-logistic-regression`](https://github.com/Ajanikehinde/bank-churn-logistic-regression) uses the same dataset and the same stratified 80/20 split (`random_state=42`), but only 7 of the 10 features (it drops `Tenure`, `HasCrCard` and `EstimatedSalary`) and also tests `class_weight='balanced'`. Its Random Forest scores ROC-AUC 0.8400 (default) and 0.8393 (balanced), against 0.8542 here, so scores are not directly comparable across the two projects.

## Contents

```
.
├── Bank_Churn_Classification_corrected.ipynb   # main analysis notebook
├── data/
│   └── Bank_churn_modelling.csv                # source dataset (10,000 rows, 14 columns)
└── README.md
```

## Dataset

The classic Kaggle "Bank Customer Churn Modelling" dataset — 10,000 customers, 14 columns, no missing values, no duplicate rows:

| Column | Type | Notes |
|---|---|---|
| RowNumber, CustomerId, Surname | — | dropped before modeling (identifiers, no predictive value) |
| CreditScore | numeric | |
| Geography | categorical | France / Germany / Spain — one-hot encoded |
| Gender | categorical | Female / Male — label-encoded |
| Age | numeric | |
| Tenure | numeric | years with the bank |
| Balance | numeric | |
| NumOfProducts | numeric | |
| HasCrCard | boolean (0/1) | |
| IsActiveMember | boolean (0/1) | |
| EstimatedSalary | numeric | |
| **Exited** | boolean (0/1) | **target** — 20.37% churn rate |

## Key caveat: the original model was fake

An earlier draft of this notebook reported a "perfect" model — 0 misclassifications, R² = 1.0. That result was invalid, caused by two stacked bugs, both fixed in this version:

1. **Target leakage.** The train/test split was built from the full dataframe (still containing the `Exited` column) instead of the feature-only dataframe, so the model was trained with the label as one of its own inputs. Confirmed by inspecting the broken model's `feature_importances_`: **100% of the importance landed on the `Exited` column itself, 0% on every real feature** — it wasn't predicting churn, it was reading it off a column with the same name.
2. **Wrong algorithm.** `Exited` is a binary label, but the original notebook used `DecisionTreeRegressor` and evaluated it with `r2_score` — a regression model and a regression metric applied to a classification problem. Isolated from the leakage bug alone, this choice produces a **negative R²** (worse than predicting the mean) and a thresholded accuracy *below* the trivial "always predict no-churn" baseline (79.6%).

The notebook's Section 8 re-runs the same feature-importance check against the corrected model as a standing sanity check — importance is now spread across genuine attributes (`Age`, `EstimatedSalary`, `Balance`, ...), with nothing dominating the way `Exited` did before.

## Methodology

1. **Data Loading & Cleaning** — load, check dtypes/nulls/duplicates, drop identifier columns.
2. **Exploratory Data Analysis** — class balance, feature distributions, correlation heatmap, churn rate by geography/gender. *(Absent from the original notebook — `matplotlib`/`seaborn`/`numpy` were imported and never used.)*
3. **Preprocessing** — `Gender` label-encoded (binary, harmless); `Geography` one-hot encoded (nominal, not ordinal — the original label-encoded it too, implying a false ordering).
4. **Train/Test Split** — 80/20, `random_state=42`, `stratify=y`. Built from the feature-only dataframe, with an explicit `'Exited' in x.columns → False` check as a guardrail against the original bug recurring.
5. **Baseline Classifier** — `DecisionTreeClassifier`, default hyperparameters.
6. **Improved Modelling** — `max_depth` selected by 5-fold cross-validation (F1) on the training set only; compared against Random Forest and Logistic Regression.
7. **Final Model Evaluation** — confusion matrix, classification report, ROC curves for all four models.
8. **Verifying There's No Leakage** — re-checks feature importances against the corrected pipeline.

## Model comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline Decision Tree (default) | 0.7865 | 0.4776 | 0.5233 | 0.4994 | 0.6885 |
| Decision Tree (max_depth=7, CV-selected) | 0.8535 | 0.6875 | 0.5135 | 0.5879 | 0.8297 |
| **Random Forest (300 trees)** | **0.8635** | **0.7724** | 0.4668 | 0.5819 | **0.8542** |
| Logistic Regression | 0.8080 | 0.5891 | 0.1867 | 0.2836 | 0.7748 |

**Random Forest is the final model**, selected on ROC-AUC — the more robust, threshold-independent metric for a risk-ranking use case like churn.

That said, it's a real tradeoff, not a clean win: the depth-7 tree has better recall (51.4% vs. 46.7%) and a near-identical F1 score (0.588 vs. 0.582). Random Forest catches fewer actual churners but is right more often when it flags one. Which model is actually "better" depends on whether the business cares more about catching at-risk customers (favor the tree) or minimizing false alarms on an expensive retention outreach (favor the forest) — this is a business decision the notebook surfaces but doesn't make for you.

Logistic Regression is clearly the weakest performer — 18.7% recall means it misses roughly 4 in 5 actual churners, consistent with a plain linear decision boundary and no class-imbalance correction.

**Top drivers (Random Forest feature importances):** `Age` (0.238), `EstimatedSalary` (0.147), `Balance` (0.143), `CreditScore` (0.143), `NumOfProducts` (0.126). No single feature dominates, which is itself part of the leakage sanity check above.

## Limitations

- **Class imbalance is not corrected.** ~20% of customers churn. None of the four models use `class_weight='balanced'`, oversampling (e.g. SMOTE), or a tuned decision threshold — the most likely lever for improving recall on the churn class.
- **Narrow hyperparameter search.** Only `max_depth` was tuned for the decision tree, over a single CV pass; Random Forest and Logistic Regression were run close to their defaults.
- **No probability calibration check.** `predict_proba` outputs feed ROC-AUC directly without verifying they're well-calibrated — matters if this is meant to rank customers by risk for a retention campaign rather than just classify them.
- **Feature set was not varied.** All 10 features are used. The companion project's Random Forest on 7 features scores lower (ROC-AUC 0.8400), which suggests `Tenure`, `HasCrCard` and `EstimatedSalary` carry useful signal for tree models; this notebook does not test that directly.
- **Random Forest vs. the depth-limited tree is unresolved by design** (see Model comparison) — pick based on your actual precision/recall priority, not just the metric that happens to be highest.

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

Verified working with:

| Package | Version |
|---|---|
| pandas | 3.0.2 |
| numpy | 2.4.4 |
| scikit-learn | 1.8.0 |
| matplotlib | 3.10.8 |
| seaborn | 0.13.2 |

(Older pandas 2.x / earlier scikit-learn versions should also work — nothing here depends on pandas 3.x-specific behavior. `seaborn>=0.12` is needed for the `errorbar=` argument used in the bar plots; on older seaborn, replace it with `ci=None`.)

## Running the notebook

```bash
jupyter notebook Bank_Churn_Classification_corrected.ipynb
```

Run **Kernel → Restart & Run All**. The notebook is verified to execute top-to-bottom with zero errors and sequential execution counts (1 → 21) against the dataset in `data/`.

