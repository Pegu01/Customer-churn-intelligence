SELECT
    DATE_TRUNC('month', t_dat) AS month,
    COUNT(*) AS total_transactions
FROM read_csv_auto('Data/Raw/HM/transactions_train.csv')
GROUP BY 1
ORDER BY 1;