    # Telco Customer Churn Analysis

> A machine learning project to predict customer churn for a telecommunications company using the `WA_Fn-UseC_-Telco-Customer-Churn` dataset.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Analysis Workflow](#analysis-workflow)
- [Key Business Insights](#key-business-insights)
- [Visualizations Produced](#visualizations-produced)
- [Suggested Code & Model Improvements](#suggested-code--model-improvements)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)
- [Results Summary](#results-summary)
- [Author](#author)

---

## Project Overview

Customer churn — the loss of clients or customers — is one of the most critical business challenges in the telecom industry. This project analyzes customer behavior using exploratory data analysis (EDA), feature engineering, and multiple supervised machine learning models to identify customers at high risk of churning. The final goal is to help the business take proactive retention actions before customers leave.

---

## Dataset

| Property | Details |
|---|---|
| **Source** | IBM Sample Dataset |
| **File** | `WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| **Rows** | 7,043 customers |
| **Target Variable** | `Churn` (Yes / No) |

**Key feature categories:**

- **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Account Info:** `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
- **Services:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`

---

## Project Structure

```
.
|-- Churn.ipynb                          # Main Jupyter Notebook (all analysis)
|-- WA_Fn-UseC_-Telco-Customer-Churn.csv # Raw dataset
|-- README.md                            # Project documentation
```

---

## Analysis Workflow

### 1. Data Loading & Cleaning

- Loaded CSV using `pandas`
- Converted `TotalCharges` from `object` to `numeric` (handled whitespace-only strings as `NaN`)
- Dropped `customerID` (non-predictive identifier)
- Imputed 11 missing `TotalCharges` values with median
- Encoded binary `Yes`/`No` columns to `1`/`0` and target `Churn` to binary

### 2. Exploratory Data Analysis (EDA)

- Overall churn rate: **~26.5%** (class imbalance noted)
- Senior citizens churn **~42%** vs ~24% for non-seniors
- Month-to-month contract customers churn **~43%** vs ~3% for two-year contracts
- Fiber optic internet users churn more (**~42%**) than DSL (~19%)
- Customers without `OnlineSecurity` / `TechSupport` churn at higher rates
- Electronic check payment method has highest churn (**~45%**)
- Higher `MonthlyCharges` correlate strongly with churn

### 3. Feature Engineering

- Created `tenure_group`: binned tenure into `0-6`, `6-12`, `12-24`, `24-48`, `48+` months
- Churn rate declines sharply with tenure: **53.3%** for 0-6 months down to **9.5%** for 48+ months
- One-hot encoded all remaining categorical variables (`InternetService`, `Contract`, `PaymentMethod`)

### 4. Class Imbalance Handling

- Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to balance training data
- Training set after SMOTE: ~equal distribution of churn vs non-churn

### 5. Modeling

Four models were trained and evaluated:

| Model | Accuracy | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | ~80% | ~0.85 | ~0.66 | ~0.55 | ~0.60 |
| Decision Tree | ~79% | ~0.72 | ~0.58 | ~0.51 | ~0.54 |
| Random Forest | ~80% | ~0.84 | ~0.65 | ~0.46 | ~0.54 |
| **XGBoost** | **~80%** | **~0.86** | **~0.66** | **~0.54** | **~0.59** |

> **Best model by ROC-AUC: XGBoost (~0.86)**

### 6. Threshold Optimization

- Default threshold of `0.5` was evaluated alongside custom threshold tuning
- **Best threshold found: `0.319`** (optimized for F1 score balance)
- At threshold `0.319`: improved recall (catching more churners) at acceptable precision cost

### 7. Hyperparameter Tuning

- Applied `GridSearchCV` / `RandomizedSearchCV` on top models
- XGBoost tuned params: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`
- Random Forest tuned params: `n_estimators`, `max_depth`, `min_samples_split`

---

## Key Business Insights

1. **Contract Type is the strongest churn predictor.** Month-to-month customers churn at 43% vs 3% for two-year contracts. Offering incentives to switch to annual/biennial contracts can significantly reduce churn.

2. **Tenure is inversely related to churn.** Customers in their first 6 months are 5x more likely to churn than those with 4+ years. Onboarding improvement and early engagement programs are critical.

3. **High monthly charges + no value-add services = highest churn risk.** Fiber optic customers paying high bills without security/tech support services churn the most.

4. **Electronic check payers churn ~45%.** Migrating these customers to auto-pay (bank transfer or credit card) could reduce churn and payment friction.

5. **Senior citizens are a high-risk segment (~42% churn).** Dedicated retention offers for seniors could protect this revenue base.

---

## Visualizations Produced

- Churn distribution (count plot)
- Churn rate by gender
- Churn by SeniorCitizen status
- Churn by contract type
- Churn by internet service type
- Churn by payment method
- Churn rate by tenure group (bar chart)
- Correlation heatmap of features
- ROC curves for all models
- Feature importance (Random Forest & XGBoost)
- Confusion matrices for each model

---

## Suggested Code & Model Improvements

### 1. Use `sklearn` Pipelines
Wrap preprocessing (imputation, scaling, encoding) and the model into a single `Pipeline` object. This prevents data leakage between train/test splits and makes deployment cleaner.

### 2. Add Feature Scaling
Logistic Regression is sensitive to feature scale. Apply `StandardScaler` or `MinMaxScaler` to numerical columns (`tenure`, `MonthlyCharges`, `TotalCharges`) to improve performance.

### 3. Use Cross-Validation
Replace single train/test split with `StratifiedKFold` (k=5 or 10) cross-validation. This gives more reliable metric estimates, especially important with class imbalance.

### 4. Improve SMOTE Application
Apply SMOTE **only inside** the cross-validation fold (not before splitting) to avoid information leakage. Use `imbalanced-learn`'s `Pipeline` for this. Also consider `SMOTE-ENN` or `SMOTE-Tomek` as cleaner alternatives.

### 5. Add Precision-Recall Curve Analysis
With class imbalance, ROC-AUC can be overly optimistic. Add **Precision-Recall AUC (PR-AUC)** as an additional evaluation metric for a more honest performance picture.

### 6. Add SHAP Explainability
Use **SHAP** (SHapley Additive exPlanations) to explain individual predictions. This is critical for business stakeholders to understand *why* a specific customer is flagged as high churn risk.

```python
import shap
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

### 7. Add Business Metrics (Revenue at Risk)
Calculate monthly revenue at risk from predicted churners:

```python
revenue_at_risk = predicted_churners['MonthlyCharges'].sum()
print(f'Monthly Revenue at Risk: ${revenue_at_risk:,.2f}')
```

### 8. Segment High-Value Churners
Filter predicted churners with `MonthlyCharges > 70` to prioritize retention spend on high-value customers, enabling targeted campaigns with maximum ROI.

### 9. Cost-Sensitive Learning
Use `class_weight='balanced'` in Logistic Regression and Random Forest as an alternative to SMOTE. For XGBoost, set:

```python
scale_pos_weight = count_negatives / count_positives  # ~2.77 for this dataset
```

### 10. Model Export & Deployment Readiness
Save the best model using `joblib` for deployment:

```python
import joblib
joblib.dump(best_model, 'churn_model.pkl')
```

Add an inference function that returns churn probability + risk tier (High / Medium / Low).

---

## Technologies Used

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualizations |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `xgboost` | Gradient boosting |
| `imbalanced-learn` | SMOTE for class imbalance |
| `Jupyter Notebook` | Development environment |

---

## How to Run

**1. Clone the repository or download the project files**

**2. Install dependencies:**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn jupyter
```

**3. Place the dataset** in `/Users/hiteshallakki/Documents/datasets/` or update the file path in the notebook.

**4. Launch Jupyter:**

```bash
jupyter notebook
```

**5. Open `Churn.ipynb`** and run all cells: `Kernel > Restart & Run All`

---

## Results Summary

| Metric | Value |
|---|---|
| **Best Model** | XGBoost |
| **ROC-AUC** | ~0.86 |
| **Optimal Threshold** | 0.319 |
| **Overall Churn Rate** | 26.5% (1,869 / 7,043 customers) |
| **Top Predictors** | Contract type, Tenure, MonthlyCharges, InternetService, OnlineSecurity |
| **Highest Risk Segment** | Month-to-month + Fiber optic + Electronic check + 0-6 month tenure (~53% churn) |

---

## Author

**Hitesh Allakki**
Data Analyst | Machine Learning | Fraud & Risk Analytics

- GitHub: [github.com/hiteshallakki](https://github.com/hiteshallakki)