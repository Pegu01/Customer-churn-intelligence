# Customer Churn Intelligence

An end-to-end customer churn prediction and retention recommendation system combining machine learning, explainable AI, revenue-at-risk analysis, and a local LLM.

---

## 📌 Project Overview

Customer churn is a major business problem because identifying customers at risk of leaving is only useful when the business can take an appropriate retention action.

This project builds an end-to-end **Customer Churn Intelligence system** that:

1. Predicts customer churn probability using XGBoost.
2. Explains individual predictions using SHAP.
3. Estimates expected revenue at risk.
4. Converts model evidence into a practical retention recommendation using a locally hosted Llama 3.2 model.
5. Provides an interactive Streamlit dashboard for customer-level analysis.

The key design principle is:

> **XGBoost predicts → SHAP explains → Llama translates.**

The LLM does not independently predict churn and is not allowed to invent customer facts.

---

# 🎯 Business Problem

Businesses often know that a customer is at risk of churn but may not know:

- Which customers should be prioritized?
- Why is a particular customer considered risky?
- How much revenue is potentially exposed?
- What retention action should be considered?

This project addresses these questions through a single analytical workflow.

---

# 🧠 System Architecture

```text
                    Customer Data
                         │
                         ▼
                Feature Engineering
                         │
                         ▼
                  XGBoost Model
                         │
                         ▼
                Churn Probability
                         │
                         ▼
                       SHAP
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       Churn Drivers          Revenue at Risk
              │                     │
              └──────────┬──────────┘
                         ▼
                  Controlled Prompt
                         │
                         ▼
                  Local Llama 3.2
                         │
                         ▼
             Retention Recommendation
                         │
                         ▼
                Streamlit Dashboard