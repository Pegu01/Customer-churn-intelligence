
````markdown
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
````

---

# 📊 Dataset

The project uses the H&M Personalized Fashion Recommendations dataset.

Main data sources include:

* `customers.csv`
* `articles.csv`
* `transactions_train.csv`

The transaction dataset contains customer purchase activity including:

* Customer ID
* Article ID
* Transaction date
* Price
* Sales channel

---

# 🔧 Feature Engineering

Customer-level features were constructed from transaction and customer information.

Examples include:

* Total orders
* Total spend
* Average order value
* Recent 90-day orders
* Prior 90-day orders
* Tenure
* Days since last purchase
* Category diversity
* Customer activity indicators
* Age
* Usage change
* Customer satisfaction
* Complaints
* Support tickets
* Payment failures
* Preferred channel
* Club membership
* Fashion news frequency

Categorical variables were encoded for machine-learning use.

The final model uses:

**22 model features.**

---

# 🤖 Machine Learning Model

## XGBoost

XGBoost was selected as the primary churn prediction model after comparing machine-learning approaches.

The model produces:

```text
P(Customer Churn)
```

which represents the predicted probability that a customer will churn.

Risk categories are defined as:

| Churn Probability | Risk   |
| ----------------: | ------ |
|             < 40% | Low    |
|      40% – 69.99% | Medium |
|             ≥ 70% | High   |

---

# 🔍 Explainable AI — SHAP

SHAP (SHapley Additive exPlanations) is used to explain individual customer predictions.

For each customer, the system identifies the features that contributed most strongly to the model's churn prediction.

A positive SHAP value means:

> The feature pushes this customer's prediction toward churn.

A negative SHAP value means:

> The feature pushes this customer's prediction away from churn.

The system displays the top five SHAP drivers.

---

# 💰 Revenue at Risk

The system estimates potential revenue exposure using:

```text
Expected Revenue at Risk
=
Customer Spend × Churn Probability
```

For example:

```text
Customer Spend = 21.26
Churn Probability = 77.83%

Revenue at Risk
= 21.26 × 0.7783
≈ 16.55
```

This is an analytical estimate and should not be interpreted as guaranteed future revenue loss.

---

# 🦙 Local LLM Recommendation Engine

The project uses:

**Ollama + Llama 3.2 3B**

for the recommendation layer.

The LLM receives controlled evidence including:

* Churn probability
* Risk level
* Customer spend
* Revenue at risk
* Top SHAP drivers
* Observed feature values
* SHAP direction

The LLM does **not** receive unrestricted raw customer data.

Its role is to translate existing analytical evidence into a concise business recommendation.

### Design Principle

```text
Machine Learning
      ↓
Prediction

     SHAP
      ↓
Explanation

     LLM 
      ↓
Business Translation
```

This reduces the risk of the LLM independently inventing churn explanations.

---

# 🖥️ Streamlit Dashboard

The project includes an interactive Streamlit dashboard.

Users can enter a Customer ID and receive:

### Customer Overview

* Churn probability
* Risk level
* Total spend
* Revenue at risk

### Explainability

* Top five churn drivers
* SHAP values
* SHAP visualization

### Recommendation

* Primary churn signal
* Recommended retention action
* Evidence-based reasoning

---

### Dashboard Preview

The Streamlit interface provides an interactive customer-level
view of:

- Churn probability
- Risk level
- Total spend
- Revenue at risk
- SHAP churn drivers
- LLM-generated retention recommendation

The dashboard is designed to turn the analytical pipeline into
a practical decision-support tool for customer retention.

---
# 📈 Example Customer Analysis

Example customer:

```text
Customer ID:
de6066796e7c487ac8f560a0054a2d33f46670665f70220e98729fbbdf7ea7ad
```

Model output:

```text
Churn Probability: 77.83%
Risk Level: High
Total Spend: 21.26
Revenue at Risk: 16.55
```

Top churn drivers included:

| Feature                  |    SHAP |
| ------------------------ | ------: |
| tenure_days              | +0.2495 |
| days_since_last_purchase | +0.2107 |
| total_orders             | +0.1984 |
| preferred_channel        | -0.1808 |
| prior_90d_orders         | +0.1284 |

The system then generates a retention recommendation using the local Llama model.

---

# 🗂️ Project Structure

```text
Customer Curn intelligence/
│
├── Data/
│   └── Raw/
│       └── HM/
│           ├── articles.csv
│           ├── customers.csv
│           └── transactions_train.csv
│
├── Notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── model_evaluation.ipynb
│   └── data/
│       └── processed/
│           ├── churn_risk_results.parquet
│           ├── modeling_table.parquet
│           └── top_500_retention_targets.csv
│
├── SRC/
│   ├── app.py
│   ├── customer_recommendation.py
│   └── llm_recommendation.py
│
├── feature_columns.pkl
├── shap_explainer.pkl
├── xgb_churn_model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Technologies Used

### Programming

* Python
* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* XGBoost

### Explainable AI

* SHAP

### Generative AI

* Ollama
* Llama 3.2 3B

### Application

* Streamlit

### Data

* Parquet
* CSV

### Development

* VS Code
* Jupyter Notebook
* Git

---

# 🚀 How to Run

## 1. Create and activate virtual environment

```bash
python -m venv venv
```

Activate on Windows:

```powershell
venv\Scripts\Activate.ps1
```

---

## 2. Install dependencies

```powershell
pip install -r requirements.txt
```

If Streamlit is not included:

```powershell
pip install streamlit
```

---

## 3. Install Ollama

Install Ollama separately and verify:

```powershell
ollama --version
```

The project was tested with Ollama 0.32.14.

---

## 4. Download the local Llama model

```powershell
ollama pull llama3.2:3b
```

Verify:

```powershell
ollama list
```

---

## 5. Run the dashboard

From the project root:

```powershell
streamlit run SRC\app.py
```

Then open:

```text
http://localhost:8501
```

---

# 🔐 Privacy and LLM Design

The recommendation engine is designed to run locally using Ollama.

Customer-level analytical information is passed to the locally running Llama model rather than a cloud-based LLM API.

The LLM is constrained to use model-generated evidence rather than independently analyzing unrestricted raw customer data.

---
# 💡 Key Analytical Insight

The system separates three distinct responsibilities:

| Component | Responsibility |
|---|---|
| XGBoost | Predict churn probability |
| SHAP | Explain the individual prediction |
| Llama 3.2 | Translate evidence into a business recommendation |

This separation is intentional. The LLM does not replace the
predictive model or the explainability layer.

> **XGBoost predicts → SHAP explains → Llama translates.**

# ⚠️ Limitations

This project has several limitations:

1. Churn labels depend on the chosen churn definition.
2. SHAP explains model behavior but does not establish causality.
3. Revenue at risk is an estimate rather than guaranteed revenue loss.
4. Retention recommendations are decision-support suggestions, not automated business decisions.
5. The model should be monitored for data drift when deployed on future customer data.
6. The LLM can still produce imperfect wording, so generated recommendations should be reviewed before operational use.
7. The H&M dataset represents a historical retail environment and may not generalize directly to every business.

---

# 🔮 Future Improvements

Potential future development includes:

* Customer segmentation
* Customer lifetime value integration
* Retention campaign optimization
* A/B testing of retention actions
* Automated retention target lists
* SHAP summary dashboards
* Model monitoring
* Data drift detection
* Batch scoring pipeline
* Campaign management integration
* Cost-sensitive churn optimization
* ROI-based retention prioritization
* Docker deployment
* Cloud deployment

---

# 💼 Business Value

The system combines predictive analytics and generative AI into a single decision-support workflow.

Instead of simply predicting:

> "This customer may churn."

the system answers:

> "How likely is the customer to churn, why does the model think so, how much revenue is potentially exposed, and what retention action could the business consider?"

This makes the project relevant to:

* Customer Analytics
* Marketing Analytics
* CRM
* Retention Strategy
* Business Intelligence
* Predictive Analytics
* Explainable AI
* Generative AI

---

# 👤 Project Focus

**Customer Churn Prediction + Explainable AI + LLM-Powered Retention Recommendations**

Built as an end-to-end Business Analytics / Marketing Analytics portfolio project.

````

