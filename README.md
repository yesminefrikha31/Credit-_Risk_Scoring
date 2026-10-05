# Credit Risk Scoring — Retail Banking

Predicting loan default from an **anonymized retail-banking dataset**, using an interpretable logistic regression pipeline.

**Stack:** Python · pandas · scikit-learn · imbalanced-learn (SMOTE) · SHAP

## Context

This project was developed as part of an academic study related to decision-making in the Tunisian banking sector.

The objective was to explore how data-driven approaches can support credit risk assessment. I developed a credit risk scoring workflow based on an anonymized retail-banking dataset to identify loan applications with a higher probability of default while keeping the model interpretable for business users.

The project covers the complete machine-learning workflow, from data quality and financial feature engineering to model evaluation, decision-threshold selection and explainability using SHAP.

> **Data:** the real dataset is confidential and is **not** included.
> `data/sample/sample.csv` is a **synthetic** file with the same schema (random values — no business conclusion can be drawn from it),

## Data dictionary

| Column | Description |
|---|---|
| `birth_date` |Applicant's date of birth (`dd/mm/yyyy`), converted to `age` in years |
| `profession` | Applicant's profession |
| `salary` |  Monthly salary |
| `credit_amount` |  Loan amount |
| `duration` | Loan duration (months) |
| `monthly_payment` |  Monthly loan repayment |
| `outstanding_credit` |  Other credit currently running (empty = none) |
| `default` |  **Target**: 1 = defaulted, 0 = healthy |


## Methodology

1. **Cleaning** — type conversions, validity rules (age, salary, duration), contradictory and true duplicates, mathematically impossible loans.
2. **Feature engineering** — debt-to-income, credit-to-annual-income, total repayment.
3. **Leak-free pipeline** — stratified split first; scaling, one-hot encoding and SMOTE are fitted on training data only.
4. **Model & validation** — regularized logistic regression, stratified 3-fold cross-validation, decision threshold chosen with Youden's J on cross-validated predictions.
5. **Interpretability** — coefficients, odds ratios, SHAP (global summary and per-client explanation).

## Project structure

```
credit-risk-scoring/
├── notebooks/01_credit_risk_scoring.ipynb   # full analysis
├── data/
│   ├── raw/                                 # confidential data (git-ignored)
│   └── sample/sample.csv                    # synthetic data, same schema
├── requirements.txt
└── README.md
```

## How to run

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/01_credit_risk_scoring.ipynb
```

Run **Kernel → Restart & Run All**. 

## Limitations

- Small dataset: metrics are noisy (see the cross-validation standard deviation).
- SMOTE and `class_weight="balanced"` both rebalance the classes; each should be tested separately.
- Single bank, single train/test split, no external validation.
