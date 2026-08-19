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