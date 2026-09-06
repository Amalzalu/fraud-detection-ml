# Fraud Detection Using Machine Learning

## Overview
A machine learning system that classifies credit card transactions as fraudulent or legitimate, built end-to-end from raw data to a deployable prediction interface. This project focuses heavily on handling extreme class imbalance correctly and choosing models based on business-relevant metrics rather than raw accuracy.

## Problem Statement
Financial fraud detection is a highly imbalanced classification problem — fraudulent transactions typically represent well under 1% of all transactions. A naive model can achieve >99% accuracy while catching zero fraud, making this a case study in why standard evaluation metrics fail and how to properly diagnose and address class imbalance.

## Dataset
- **Source:** [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) (ULB, via Kaggle)
- 283,726 transactions after cleaning (1,081 duplicate rows removed)
- 30 features: `Time`, `Amount`, and 28 PCA-transformed anonymized features (V1–V28)
- Target: `Class` (0 = Legitimate, 1 = Fraud)
- **Class distribution:** 99.83% Legitimate vs 0.167% Fraud (473 fraud cases total)

## Technologies Used
- Python, pandas, numpy
- scikit-learn (modeling, preprocessing, evaluation)
- imbalanced-learn (SMOTE)
- matplotlib, seaborn (visualization)
- Streamlit (deployment demo)
- joblib (model persistence)

## Machine Learning Pipeline
Data Loading → Cleaning → EDA → Feature/Target Split → Train/Test Split (stratified) 
→ Scaling → SMOTE (train only) → Model Training (Logistic Regression, KNN, LDA) 
→ Evaluation → Model Selection → Deployment

## Data Preprocessing
- Removed 1,081 exact duplicate transactions to prevent train/test leakage
- Verified no missing values
- Investigated zero-amount transactions (1,808 found, 25 fraudulent) — kept, since these represent real "card testing" fraud patterns rather than data errors
- Deliberately did NOT remove outliers, since fraud transactions are inherently statistical outliers

## Exploratory Data Analysis
[Insert your 5 key plots here as images — export them from your notebook using `plt.savefig('../images/plot_name.png')`]
- Class distribution confirms severe imbalance
- Fraud transactions skew toward smaller amounts
- Fraud and legitimate transactions show different `Time` distributions, justifying keeping `Time` as a feature
- Correlation analysis identified V2, V4, V11 (and others) as most correlated with fraud

## Class Imbalance
- Fraud rate: 0.167% (roughly 1 in 600 transactions)
- Addressed using **SMOTE**, applied only to training data after the train/test split, to avoid data leakage
- Also considered `class_weight='balanced'` as a lower-cost alternative

## Models Used

### Logistic Regression
Baseline linear classifier using a sigmoid function to output fraud probability. Trained on SMOTE-resampled data.

### K-Nearest Neighbors (KNN)
Instance-based classifier using distance-based voting (K=5). Requires feature scaling due to its reliance on distance calculations.

### Linear Discriminant Analysis (LDA)
Generative-adjacent linear classifier assuming Gaussian class distributions. Included to test whether this assumption holds on this dataset (it does not hold well, given `Amount`'s skewed distribution).

## Evaluation Metrics
Accuracy alone is misleading on this dataset — a model predicting "legitimate" for every transaction would score 99.83% accuracy while catching zero fraud. Evaluation instead centers on:
- **Precision:** Of transactions flagged as fraud, how many were correct?
- **Recall:** Of all actual fraud, how many were caught?
- **F1-score:** Harmonic mean balancing precision and recall
- **ROC-AUC / PR-AUC:** Threshold-independent measures of discriminative power (PR-AUC is more informative here, given the extreme imbalance)

## Results

| Model | Accuracy | Precision (Fraud) | Recall (Fraud) | F1 (Fraud) | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9737 | 0.0530 | 0.8737 | 0.1000 | 0.9626 | 0.6750 |
| KNN | 0.9980 | 0.4425 | 0.8105 | 0.5725 | 0.9100 | 0.5388 |
| LDA | 0.9889 | 0.1092 | 0.7895 | 0.1918 | 0.9514 | 0.2857 |

## Confusion Matrices
[Insert your normalized confusion matrix heatmap image here]

## ROC Curve
[Insert your ROC curve comparison plot here]

## Key Findings
- **Accuracy is a misleading metric on imbalanced fraud data** — all three models exceed 97% accuracy, but their real fraud-catching ability varies drastically.
- **Logistic Regression achieved the highest recall (87.4%) and the strongest underlying discriminative power (ROC-AUC 0.963, PR-AUC 0.675)**, making it the preferred model when minimizing missed fraud is the priority — at the cost of a high false-positive rate (5.3% precision) that would require threshold tuning for real-world deployment.
- **KNN achieved the best F1-score (0.5725) and precision (44.3%)** with a more balanced but more conservative trade-off, at the cost of significantly slower prediction time (~25 seconds for 56,746 transactions) due to its lazy-learning nature.
- **LDA underperformed both other models**, likely due to its Gaussian distribution assumption being violated by the right-skewed `Amount` feature.
- **ROC-AUC can overstate performance on highly imbalanced data** — LDA's ROC-AUC (0.951) looked competitive with Logistic Regression's, but its PR-AUC (0.286) revealed it was actually the weakest model.
- **Final model selected: Logistic Regression**, prioritizing recall given the asymmetric cost of missed fraud (financial loss) vs false alarms (customer friction), with threshold tuning identified as a recommended next step.

## How to Run

1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Download the dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place `creditcard.csv` in `data/`
6. Run notebooks in `notebooks/` in order to reproduce the analysis and train models
7. Launch the demo app: `cd app && streamlit run app.py`

## Project Structure
\```
fraud-detection-ml/
├── data/               # Dataset (not committed - see How to Run)
├── notebooks/          # EDA and model development notebooks
├── src/                # Reusable scripts
├── models/             # Saved model and scaler (.pkl)
├── images/             # Saved plots for documentation
├── app/                # Streamlit application
├── requirements.txt
└── README.md
\```

## Future Improvements
- Tune the Logistic Regression decision threshold to improve precision without excessive recall loss
- Try additional models (Random Forest, XGBoost) for comparison
- Explore engineered time-based features (e.g., transaction hour, if raw timestamps were available)
- Address KNN's prediction latency for real-time deployment feasibility
- Deploy behind a proper API rather than a local Streamlit demo

## Author
Ali Miyan MC 
www.linkedin.com/in/ali-miyan
https://github.com/Ali-miyan-mc2004
