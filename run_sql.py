import duckdb

con = duckdb.connect()

with open("SQL/monthly_transactions.sql", "r", encoding="utf-8") as f:
    sql = f.read()

result = con.execute(sql)

print(result.fetchdf())

con.close()