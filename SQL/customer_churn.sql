DESCRIBE
SELECT *
FROM read_csv_auto('Data/Raw/HM/transactions_train.csv');

SELECT
    MIN(t_dat) AS earliest_transaction,
    MAX(t_dat) AS latest_transaction
FROM read_csv_auto('Data/Raw/HM/transactions_train.csv');

SELECT
    COUNT(DISTINCT customer_id) AS transacting_customers
FROM read_csv_auto('Data/Raw/HM/transactions_train.csv');

---------------------------------------------------
WITH feature_window AS (
    SELECT
        customer_id,
        MAX(t_dat) AS last_purchase_in_feature_window
    FROM read_csv_auto('Data/Raw/HM/transactions_train.csv')
    WHERE t_dat <= '2020-05-25'
    GROUP BY customer_id
)

SELECT *
FROM feature_window
LIMIT 10;

---------------------------------------------
WITH label_window AS (
    SELECT
        customer_id
    FROM read_csv_auto('Data/Raw/HM/transactions_train.csv')
    WHERE t_dat BETWEEN '2020-05-26' AND '2020-09-22'
    GROUP BY customer_id
)

SELECT *
FROM label_window
LIMIT 10;

-----------------------------------------------

WITH feature_window AS (
    SELECT
        customer_id,
        MAX(t_dat) AS last_purchase_in_feature_window
    FROM read_csv_auto('Data/Raw/HM/transactions_train.csv')
    WHERE t_dat <= '2020-05-25'
    GROUP BY customer_id
),

label_window AS (
    SELECT
        customer_id
    FROM read_csv_auto('Data/Raw/HM/transactions_train.csv')
    WHERE t_dat BETWEEN '2020-05-26' AND '2020-09-22'
    GROUP BY customer_id
)

SELECT
    f.customer_id,
    f.last_purchase_in_feature_window,
    l.customer_id AS label_customer_id
FROM feature_window f
LEFT JOIN label_window l
    ON f.customer_id = l.customer_id
LIMIT 20;

--------------------------------------

WITH feature_window AS (
    SELECT
        customer_id,
        MAX(t_dat) AS last_purchase_in_feature_window
    FROM read_csv_auto('Data/Raw/HM/transactions_train.csv')
    WHERE t_dat <= '2020-05-25'
    GROUP BY customer_id
),

label_window AS (
    SELECT
        customer_id
    FROM read_csv_auto('Data/Raw/HM/transactions_train.csv')
    WHERE t_dat BETWEEN '2020-05-26' AND '2020-09-22'
    GROUP BY customer_id
),

customer_churn AS (
    SELECT
        f.customer_id,
        f.last_purchase_in_feature_window,

        CASE
            WHEN l.customer_id IS NOT NULL THEN 1
            ELSE 0
        END AS purchased_in_label_window,

        CASE
            WHEN l.customer_id IS NULL THEN 1
            ELSE 0
        END AS churn

    FROM feature_window f
    LEFT JOIN label_window l
        ON f.customer_id = l.customer_id
)

SELECT
    COUNT(*) AS total_customers,
    SUM(churn) AS churned_customers,
    COUNT(*) - SUM(churn) AS retained_customers,
    ROUND(AVG(churn) * 100, 2) AS overall_churn_rate_percent
FROM customer_churn;

----------------------------------------------------------
SELECT
    COUNT(DISTINCT customer_id) AS feature_window_customers
FROM read_csv_auto('Data/Raw/HM/transactions_train.csv')
WHERE t_dat <= '2020-05-25';