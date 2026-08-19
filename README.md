# Customer Churn Intelligence

An end-to-end customer churn analysis and prediction project using Python, SQL, machine learning, and dashboarding.

## Project Structure

```text
customer-churn-intelligence/
│
├── data/
│   ├── raw/
│   │   └── hm/
│   ├── processed/
│   └── synthetic/
├── notebooks/
├── sql/
├── src/
├── dashboard/
├── reports/
├── README.md
├── requirements.txt
└── .gitignore

COVID-period data limitation: The transaction dataset ends in September 2020, limiting the ability to assess longer-term post-pandemic customer behavior. Monthly transaction volume does not show a sustained decline beginning in March–April 2020. Transactions increased from 1,047,752 in March 2020 to 1,764,507 in June 2020 before declining to 798,269 in September 2020. Therefore, the dataset does not support attributing the observed customer churn rate directly to a sustained COVID-period transaction collapse. The lack of data beyond September 2020 nevertheless prevents assessment of longer-term recovery and post-pandemic retention patterns.


XGBoost marginally outperforms Logistic Regression on both ROC-AUC and PR-AUC. Therefore, XGBoost is selected as the primary predictive model, while Logistic Regression serves as an interpretable baseline.Why XGBoost is still our winner : PR-AUC is particularly important here because we're interested in identifying customers likely to churn.

XGBoost:
PR-AUC = 0.8759
versus Logistic Regression:
PR-AUC = 0.8696
So XGBoost has the better ranking performance on the test set.

H&M's price field is normalized [0,1] to anonymize real prices — confirmed via independent research, not officially disclosed by H&M. The 590x factor below is a community-derived approximation (Kaggle discussion #310496), NOT verified ground truth.