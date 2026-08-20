# Check Customer Spend

import pandas as pd

customer_id = (
    "de6066796e7c487ac8f560a0054a2d33f46670665f70220e98729fbbdf7ea7ad"
)

file_path = (
    "Data/Raw/HM/transactions_train.csv"
)

total_spend = 0.0
transaction_count = 0

print("Reading transactions...")

for chunk in pd.read_csv(
    file_path,
    usecols=["customer_id", "price"],
    chunksize=200_000
):

    customer_transactions = chunk[
        chunk["customer_id"] == customer_id
    ]

    if not customer_transactions.empty:

        total_spend += customer_transactions["price"].sum()

        transaction_count += len(
            customer_transactions
        )


print()
print("=" * 60)
print("CUSTOMER SPEND CHECK")
print("=" * 60)

print("Customer ID:", customer_id)

print(
    "Number of transactions:",
    transaction_count
)

print(
    "Raw normalized spend:",
    total_spend
)

print(
    "Spend × 100:",
    total_spend * 100
)

print("=" * 60)