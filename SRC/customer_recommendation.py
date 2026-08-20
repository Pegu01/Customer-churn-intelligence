# Customer Recommendation System

import os
import joblib
import pandas as pd

from llm_recommendation import generate_llm_recommendation

# --------------------------------------------------
# Calculate actual customer spend
# --------------------------------------------------

def get_actual_customer_spend(customer_id):

    transaction_path = os.path.join(
        BASE_DIR,
        "Data",
        "Raw",
        "HM",
        "transactions_train.csv"
    )

    total_spend = 0.0

    # Read the large transaction file in chunks
    for chunk in pd.read_csv(
        transaction_path,
        usecols=["customer_id", "price"],
        chunksize=200_000
    ):

        customer_transactions = chunk[
            chunk["customer_id"] == customer_id
        ]

        if not customer_transactions.empty:
            total_spend += customer_transactions["price"].sum()

    return float(total_spend)

# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)
# --------------------------------------------------
# Calculate actual customer spend
# --------------------------------------------------

def get_actual_customer_spend(customer_id):

    transaction_path = os.path.join(
        BASE_DIR,
        "Data",
        "Raw",
        "HM",
        "transactions_train.csv"
    )

    total_spend = 0.0

    # Read the large transaction file in chunks
    for chunk in pd.read_csv(
        transaction_path,
        usecols=["customer_id", "price"],
        chunksize=200_000
    ):

        customer_transactions = chunk[
            chunk["customer_id"] == customer_id
        ]

        if not customer_transactions.empty:
            total_spend += customer_transactions["price"].sum()

    return float(total_spend)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "xgb_churn_model.pkl"
)

SHAP_PATH = os.path.join(
    BASE_DIR,
    "shap_explainer.pkl"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "feature_columns.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "Notebooks",
    "data",
    "processed",
    "modeling_table.parquet"
)


# --------------------------------------------------
# Load saved ML components
# --------------------------------------------------

xgb_model = joblib.load(MODEL_PATH)

explainer = joblib.load(SHAP_PATH)

feature_columns = joblib.load(FEATURE_PATH)


# --------------------------------------------------
# Load processed customer data
# --------------------------------------------------

df = pd.read_parquet(DATA_PATH)

# --------------------------------------------------
# Create the same one-hot encoded columns
# --------------------------------------------------

df = pd.get_dummies(
    df,
    columns=[
        "club_member_status",
        "fashion_news_frequency"
    ],
    dtype=int
)

print("Feature engineering completed.")
print("\nEncoded columns:")
print(df.columns.tolist())


print("XGBoost model loaded.")
print("SHAP explainer loaded.")
print("Feature list loaded.")
print("Customer data loaded.")

print("Number of model features:", len(feature_columns))
print("Number of customers:", len(df))


print("XGBoost model loaded.")
print("SHAP explainer loaded.")
print("Feature list loaded.")
print("Number of features:", len(feature_columns))

# --------------------------------------------------
# Load processed customer data
# --------------------------------------------------

DATA_PATH = os.path.join(
    BASE_DIR,
    "Notebooks",
    "data",
    "processed",
    "modeling_table.parquet"
)

# --------------------------------------------------
# Align customer data with the trained model
# --------------------------------------------------

# The modeling table already contains the encoded
# club_member_status and fashion_news_frequency columns.

# Add any model features that are missing
for column in feature_columns:
    if column not in df.columns:
        df[column] = 0

# Keep only the exact features used during training
X_all = df[feature_columns].copy()

print("Model input created.")
print("X_all shape:", X_all.shape)


# --------------------------------------------------
# Test one real customer
# --------------------------------------------------

# --------------------------------------------------
# Select a customer for testing
# --------------------------------------------------

customer_id = df["customer_id"].iloc[1000]

print()
print("=" * 70)
print("TEST CUSTOMER")
print("=" * 70)
print("Customer ID:", customer_id)

# Find the customer in the original data
customer_rows = df[
    df["customer_id"] == customer_id
]

if customer_rows.empty:
    raise ValueError(
        f"Customer ID not found: {customer_id}"
    )

# Get the row position
customer_index = customer_rows.index[0]

print()
print("=" * 70)
print("CUSTOMER FOUND")
print("=" * 70)
print("Customer ID:", customer_id)
print("Data row index:", customer_index)

# --------------------------------------------------
# Calculate churn probability
# --------------------------------------------------

customer_features = X_all.loc[[customer_index]]

churn_probability = xgb_model.predict_proba(
    customer_features
)[0, 1]

print()
print("=" * 70)
print("CHURN PREDICTION")
print("=" * 70)
print(
    "Churn Probability:",
    f"{churn_probability:.2%}"
)

# --------------------------------------------------
# Calculate SHAP values
# --------------------------------------------------

customer_shap = explainer.shap_values(
    customer_features
)

# Get SHAP values for this customer
customer_shap = customer_shap[0]

print()
print("=" * 70)
print("SHAP CALCULATION")
print("=" * 70)

print("SHAP values calculated.")
print("Number of SHAP values:", len(customer_shap))

# --------------------------------------------------
# Identify top 5 SHAP drivers
# --------------------------------------------------

shap_explanation = pd.DataFrame({
    "Feature": feature_columns,
    "Value": customer_features.iloc[0].values,
    "SHAP": customer_shap
})

shap_explanation["Absolute_SHAP"] = (
    shap_explanation["SHAP"].abs()
)

shap_explanation = shap_explanation.sort_values(
    "Absolute_SHAP",
    ascending=False
)

top_5_drivers = shap_explanation.head(5)

print()
print("=" * 70)
print("TOP 5 CHURN DRIVERS")
print("=" * 70)

print(
    top_5_drivers[
        ["Feature", "Value", "SHAP", "Absolute_SHAP"]
    ].to_string(index=False)
)

# --------------------------------------------------
# Prepare SHAP drivers for LLM
# --------------------------------------------------

top_shap_drivers = []

for _, row in top_5_drivers.iterrows():

    feature = row["Feature"]
    value = row["Value"]
    shap_value = float(row["SHAP"])

    # Convert preferred_channel code to readable name
    if feature == "preferred_channel":

        preferred_channel_map = {
            0: "Email",
            1: "App",
            2: "SMS"
        }

        value = preferred_channel_map.get(
            int(value),
            "Unknown"
        )

    # Determine SHAP direction
    if shap_value > 0:
        direction = "pushes toward churn"
    else:
        direction = "pushes away from churn"

    top_shap_drivers.append({
        "feature": feature,
        "value": value,
        "shap": shap_value,
        "direction": direction
    })


print()
print("=" * 70)
print("SHAP DATA PREPARED FOR LLM")
print("=" * 70)

for driver in top_shap_drivers:
    print(driver)

# --------------------------------------------------
# Determine risk level
# --------------------------------------------------

if churn_probability >= 0.70:
    risk_level = "High"
elif churn_probability >= 0.40:
    risk_level = "Medium"
else:
    risk_level = "Low"

print()
print("=" * 70)
print("CUSTOMER RISK")
print("=" * 70)
print("Risk Level:", risk_level)

# --------------------------------------------------
# Get customer total spend
# --------------------------------------------------

total_spend = get_actual_customer_spend(
    customer_id
)

# Convert H&M normalized price into project monetary units
total_spend = total_spend * 100

if total_spend <= 0:
    raise ValueError(
        "No transaction spend found for this customer."
    )

print()
print("=" * 70)
print("SPEND CHECK")
print("=" * 70)

customer_raw_row = df[
    df["customer_id"] == customer_id
]

print(
    customer_raw_row[
        ["customer_id", "total_orders", "total_spend"]
    ].to_string(index=False)
)

print("Total Spend:", f"{total_spend:,.2f}")

# --------------------------------------------------
# Calculate expected revenue at risk
# --------------------------------------------------

revenue_at_risk = (
    total_spend * churn_probability
)

print(
    "Expected Revenue at Risk:",
    f"{revenue_at_risk:,.2f}"
)

# --------------------------------------------------
# Generate LLM recommendation
# --------------------------------------------------

recommendation = generate_llm_recommendation(
    customer_id=customer_id,
    churn_probability=churn_probability,
    risk_level=risk_level,
    total_spend=total_spend,
    revenue_at_risk=revenue_at_risk,
    top_shap_drivers=top_shap_drivers
)

# --------------------------------------------------
# Display final recommendation
# --------------------------------------------------

print()
print("=" * 70)
print("FINAL CUSTOMER CHURN RECOMMENDATION")
print("=" * 70)

print("Customer ID:", customer_id)

print(
    "Churn Probability:",
    f"{churn_probability:.2%}"
)

print("Risk Level:", risk_level)

print(
    "Total Spend:",
    f"{total_spend:,.2f}"
)

print(
    "Expected Revenue at Risk:",
    f"{revenue_at_risk:,.2f}"
)

print()
print("RECOMMENDATION")
print("-" * 70)

print(recommendation)

print("=" * 70)