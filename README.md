# Fraud Detection Modeling: Capital One Credit Card Transactions

## Project Overview

This project analyzes credit card transaction data and builds a fraud detection model. The dataset contains line-delimited JSON transaction records from 2016. The main goals are to:

1. Load and clean the transaction data.
2. Explore transaction patterns and fraud behavior.
3. Identify duplicate-like transactions, including reversals and multi-swipes.
4. Build predictive models to classify fraudulent transactions.
5. Evaluate model performance using appropriate metrics for imbalanced data.

The dataset is highly imbalanced: only about 1.58% of transactions are fraudulent. Because of this, the project focuses on recall and F1-score instead of accuracy alone.

---

## Repository Structure

```text
fraud-detection-modeling/
│
├── dataset/
│   ├── transactions.txt              # Raw line-delimited JSON data
│   ├── transactions_clean.csv        # Cleaned transaction data
│   └── transactions_eda.csv          # Data after duplicate flags are added
│
├── python-script/
│   └── load_and_clean.ipynb                # Notebook for loading and cleaning data
│   └── exploratory_data_analysis.ipynb     # Notebook for exploratory data analysis and duplicate detection
│   └── modeling.ipynb                      # Notebook modeling
│
├── README.md
└── .gitignore
```

Note: The dataset files are excluded from GitHub using `.gitignore` because they are very large.

---

## Data Loading and Cleaning

The dataset contains 786,363 transactions and 29 original variables. The transaction dates range from January 2016 to December 2016.

During cleaning, columns with only one unique value are removed because they do not provide predictive value:

- `echoBuffer`
- `merchantCity`
- `merchantState`
- `merchantZip`
- `posOnPremises`
- `recurringAuthInd`

Empty strings are replaced with `NaN`, then filled with `"Unknown"` for selected categorical columns:

- `acqCountry`
- `merchantCountryCode`
- `posEntryMode`
- `posConditionCode`
- `transactionType`

---

## Exploratory Data Analysis

The target variable is `isFraud`.

Out of 786,363 transactions, 12,417 are fraud transactions. This means the fraud rate is about 1.58%, so the dataset is highly imbalanced.

Important findings:

- Fraudulent transactions have a higher average transaction amount than non-fraudulent transactions.
- `transactionAmount` has a stronger relationship with fraud than balance-related variables.
- `availableMoney` and `currentBalance` have similar distributions for fraud and non-fraud transactions.
- Some categorical fields, such as `merchantCategoryCode`, `posEntryMode`, and `posConditionCode`, show different fraud rates across categories.

---

## Duplicate Transaction Detection

The project identifies two types of duplicate-like transactions:

### 1. Reversal Transactions

A reversal transaction is a transaction where a purchase is followed by a reversal. The logic checks whether a reversal has a matching purchase with the same:

- `customerId`
- `merchantName`
- `transactionAmount`

Two approaches are used:

1. Check whether the reversal is directly after a matching purchase in the sorted data.
2. For remaining reversals, search for a matching purchase with the same customer, merchant, and amount.

After identifying reversal duplicates, a new column is added:

```python
df["isDuplicate"] = False
```

Rows identified as reversal duplicates are marked as:

```python
isDuplicate = True
```

### 2. Multi-Swipe Transactions

A multi-swipe transaction is defined as a repeated purchase with the same:

- `customerId`
- `merchantName`
- `transactionAmount`

within 2 minutes.

Only non-duplicate purchase transactions are considered. The first transaction is treated as normal, and only the repeated transaction is marked as a multi-swipe.

A new column is added:

```python
df["isMultiSwipe"] = False
```

Rows identified as multi-swipes are marked as:

```python
isMultiSwipe = True
```

These rows are also marked as duplicate in the general duplicate flag:

```python
isDuplicate = True
```

Findings:

- Multi-swipe duplicate transactions found: 4,893
- Total multi-swipe amount: $738,267.73

---

## Modeling Methodology

The goal is to predict whether a transaction is fraudulent using the `isFraud` label.

Because the original dataset is highly imbalanced, the modeling dataset is adjusted by sampling:

- All fraud transactions
- 60,000 non-fraud transactions

This creates a more balanced training dataset so the model can better learn fraud patterns.

### Feature Engineering

Several categorical variables are transformed based on fraud likelihood patterns observed during EDA:

- `merchantCategoryCode`
- `posEntryMode`
- `posConditionCode`
- `acqCountry`
- `merchantCountryCode`

`transactionAmount` is also grouped into amount ranges.

Some columns are removed before modeling, including:

- `customerId`
- timestamp columns
- `merchantName`
- `availableMoney`
- `currentBalance`
- `expirationDateKeyInMatch`

Categorical variables are one-hot encoded before modeling.

---

## Models Tested

### 1. Logistic Regression

Logistic regression is used as a baseline model because it is simple and interpretable.

### 2. XGBoost with Default Parameters

XGBoost is tested because it can model nonlinear relationships and interactions between features.

Default XGBoost result:

```text
Accuracy: 83.24%
Precision: 52.23%
Recall: 4.80%
F1-score: 8.79%
```

Although the accuracy and precision are reasonable, the recall is very low. This means the model misses most fraud cases.

### 3. XGBoost with SMOTE

SMOTE is applied to oversample the fraud class in the training data.

Result:

```text
Accuracy: 65.72%
Precision: 28.09%
Recall: 66.48%
F1-score: 39.49%
```

This model has lower accuracy, but much higher recall. For fraud detection, this is more useful because catching fraud is more important than maximizing accuracy.

### 4. XGBoost with GridSearchCV

GridSearchCV is used to tune XGBoost hyperparameters using F1-score as the scoring metric.

GridSearchCV result:

```text
Confusion Matrix:
[[7670 4377]
 [ 763 1674]]

Accuracy: 64.51%
Precision: 27.66%
Recall: 68.69%
F1-score: 39.44%
```

This model has the highest recall among the models tested, but the F1-score is similar to the SMOTE model.

---

## Model Evaluation

Because the dataset is imbalanced, accuracy alone is not a good metric. A model can get high accuracy by predicting almost everything as non-fraud.

The main metrics used are:

- **Precision**: Of the transactions predicted as fraud, how many are actually fraud?
- **Recall**: Of the actual fraud transactions, how many did the model catch?
- **F1-score**: Balance between precision and recall.
- **Confusion matrix**: Shows true positives, false positives, true negatives, and false negatives.
- **ROC curve / AUC**: Measures ranking ability across thresholds.

For this project, recall is especially important because false negatives mean fraudulent transactions were missed.

---

## Feature Importance

Feature importance is analyzed using:

1. XGBoost built-in feature importance
2. SHAP values

SHAP is used to better explain how each feature contributes to model predictions.

### Important Insights

The SHAP results show that the model relied most on transaction amount ranges, card-present status, merchant category, POS entry mode, and credit limit.

`transactionAmount_0-50` was the most influential feature, suggesting that small transaction amounts played an important role in fraud prediction. Other amount buckets, such as `51-100`, `101-150`, `251-500`, and `501-750`, were also important, showing that fraud patterns vary by spending range.

`cardPresent_False` was another strong feature, meaning card-not-present transactions had a major impact on predictions. This makes sense because online or remote transactions can be riskier when stolen card information is used.

`merchantCategoryCode` also contributed strongly, showing that fraud risk differs across merchant categories. Credit limit and POS entry mode provided additional context, but they were generally less important than transaction amount, card-present status, and merchant category.

Overall, the SHAP results support the EDA findings that fraud behavior is strongly related to transaction size, transaction channel, and merchant type.

One limitation is that SHAP explains the model's learned patterns, not necessarily real-world causality. For example, a feature having a high SHAP value means it strongly influenced the model prediction, but it does not prove that the feature directly causes fraud. Also, because some categorical variables were manually grouped based on fraud rates, future work should ensure these mappings are created using only the training set to avoid data leakage.

---

## Key Conclusions

- The dataset is highly imbalanced, with fraud making up only about 1.58% of transactions.
- Fraudulent transactions tend to have higher transaction amounts than non-fraudulent transactions.
- Duplicate-like transactions can be detected using rule-based logic for reversals and multi-swipes.
- Default models perform poorly on recall because fraud is rare.
- XGBoost with SMOTE significantly improves recall.
- GridSearchCV slightly improves recall but does not greatly improve F1-score compared with the SMOTE model.
- In fraud detection, a model with lower accuracy but much higher recall may be preferred because missing fraud is costly.

---

## Limitations

There are several limitations in this project:

- The non-fraud class is undersampled before modeling, so the model is trained on a distribution different from the original dataset.
- Some feature mappings are based on observed fraud rates, which may introduce leakage if not carefully done only on the training data.
- The multi-swipe rule uses a 2-minute window, which may miss some duplicates or flag legitimate repeated purchases.
- The model does not yet use customer-level behavioral features, such as average spending amount, transaction frequency, or merchant history.
- The final model threshold is still the default 0.5 threshold. A better threshold could be selected based on business cost.

---

## Future Improvements

With more time, I would:

- Split the train and test sets before any resampling to preserve a realistic test distribution.
- Create customer-level historical features, such as average transaction amount and number of transactions in the past day.
- Tune the fraud probability threshold to balance precision and recall.
- Try additional models such as Random Forest, LightGBM, and CatBoost.
- Use cross-validation with SMOTE inside a pipeline to avoid data leakage.
- Compare ROC-AUC with PR-AUC, since PR-AUC is more informative for imbalanced fraud detection.
- Improve duplicate detection by testing multiple time windows for multi-swipes.
- Add more explanation using SHAP dependence plots.

---

## Author

Quyen Le