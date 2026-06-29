# 📊 Telco Customer Churn Prediction

## Overview

This project predicts whether a telecom customer is likely to churn (leave the company's service) using machine learning. The goal wasn't only to build a predictive model, but to understand the complete ML workflow — from data cleaning and EDA to model evaluation and business interpretation.

## Problem Statement

Customer churn is one of the biggest challenges for subscription-based businesses. Acquiring a new customer typically costs more than retaining an existing one, so being able to identify customers who are likely to leave allows companies to take proactive retention measures, reducing revenue loss and improving customer satisfaction.

This project builds and compares classification models to predict customer churn based on demographics, account information, and service usage.

## Dataset

- **Source:** [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle)
- **Rows:** 7,043
- **Features:** 21
- **Target variable:** `Churn`

The dataset includes:
- Customer demographics
- Contract type
- Internet service
- Monthly and total charges
- Tenure
- Payment method
- Additional telecom services (online security, tech support, streaming, etc.)

## Project Workflow

### 1. Data Exploration
- Explored dataset structure and data types
- Checked class distribution (26.6% churn rate overall)
- Identified categorical vs. numerical features

### 2. Data Cleaning
- Removed the non-predictive `customerID` column
- Converted `TotalCharges` from object to numeric
- Handled resulting missing values
- Verified dataset consistency

### 3. Exploratory Data Analysis (EDA)
Visualized key churn drivers before modeling:
- **Contract type** — month-to-month customers churn far more than annual/two-year contract customers
- **Tenure** — newer customers are at higher risk of churning; loyalty builds over time
- **Monthly charges** — higher-paying customers show increased churn likelihood
- **Internet service type** — Fiber optic customers churn at ~41.9%, compared to ~19% for DSL and ~7.4% for customers with no internet service

### 4. Feature Engineering
- One-hot encoding using `pd.get_dummies()`
- Train-test split (performed **before** scaling, to avoid data leakage)
- `StandardScaler` applied for Logistic Regression (fit on training data only, then applied to test data)

### 5. Models Implemented

**Logistic Regression** — used as the baseline model.

**Balanced Logistic Regression** — used `class_weight="balanced"` to improve recall for the minority class (customers who churn).

**Random Forest Classifier** — tree-based ensemble model, also used to extract feature importance.

## Evaluation Metrics

Rather than relying on accuracy alone, models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

| Model | Accuracy | Recall (Churn) | Notes |
|---|---|---|---|
| Logistic Regression (baseline) | 78.82% | 52% | Good baseline, but misses nearly half of actual churners. |
| Logistic Regression (scaled) | 78.75% | 52% | Scaling had minimal impact on Logistic Regression performance. |
| Logistic Regression (balanced) | 73.13% | 79% | Lower accuracy, but far better at catching customers likely to churn. |
| Random Forest | 78.54% | 48% | Comparable accuracy to baseline, with useful feature importance, but lowest recall. |

## Key Learning

One of the biggest insights from this project was understanding that **higher accuracy does not always mean a better model.**

The Balanced Logistic Regression had lower overall accuracy than the other models, but its recall for the churn class jumped from 52% to 79%. In a churn problem, missing a customer who's about to leave is usually far more costly than a false alarm — so recall matters more than raw accuracy here. This reinforced that evaluation metrics should be chosen based on the business objective, not optimized blindly.

## Feature Importance

Random Forest identified the following as the most influential features in predicting churn:

1. **TotalCharges**
2. **MonthlyCharges**
3. **Tenure**
4. **InternetService (Fiber optic)**
5. **PaymentMethod (Electronic check)**
6. **OnlineSecurity**
7. **Contract (Two year)**

This lines up closely with the EDA findings — contract length, tenure, charges, and internet service type all emerged as the strongest churn signals from two independent analyses.

## Business Recommendations

- Offer incentives to convert month-to-month customers into longer-term contracts.
- Launch a "First 90 Days" engagement program to improve retention among new customers.
- Review pricing and offer personalized deals for high-monthly-charge customers.
- Investigate and improve the Fiber optic service experience, given its disproportionately high churn rate.
- Use the trained model to flag high-risk customers so retention teams can intervene proactively.

## Technologies Used

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Google Colab

## Skills Demonstrated

Data Cleaning · Exploratory Data Analysis · Feature Engineering · One-Hot Encoding · Train-Test Splitting & Data Leakage Prevention · StandardScaler · Logistic Regression · Random Forest · Model Evaluation · Feature Importance · Business Interpretation

## Future Improvements

- Hyperparameter tuning (GridSearchCV/RandomizedSearchCV)
- Cross-validation for more robust performance estimates
- ROC-AUC analysis
- Model deployment using Streamlit
- Explainability using SHAP

## Project Outcome

This project helped me move beyond simply training a model. It taught me how to:
- Understand and frame a real business problem
- Build and compare multiple machine learning models
- Evaluate models using metrics appropriate to the problem, not just accuracy
- Interpret model predictions and feature importance
- Connect technical results to real-world business decisions

## How to Run

1. Clone this repository
   git clone https://github.com/kirti-050/telco-customer-churn-prediction
2. Install dependencies
   pip install -r requirements.txt
3. Open `telco_customer_churn_analysis.ipynb` in Jupyter or Google Colab and run all cells

## Author

Kirti — [LinkedIn](www.linkedin.com/in/kirti-srivastava-16a7a3290) | [GitHub](#)
## How to Run

1. Clone this repository
