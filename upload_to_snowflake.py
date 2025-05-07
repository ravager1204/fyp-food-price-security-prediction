from sqlalchemy import create_engine
import pandas as pd

# Sample data
df = pd.DataFrame({
    "date": ["2025-03-01", "2025-03-02"],
    "state": ["Selangor", "Johor"],
    "item": ["RICE", "CHILI"],
    "predicted_price_change": [2.1, -1.3],
    "predicted_fsi": [78.2, 65.0]
})

# Proper connection string for new Snowflake web UI format
conn_str = (
    "snowflake://fahimmunirzaki:Zaq12wsx123456789%@/"
    "?account=fq41450.uzyefre"
    "&warehouse=COMPUTE_WH"
    "&database=FYP_DB"
    "&schema=PUBLIC"
    "&role=ACCOUNTADMIN"
)


# Connect and upload
engine = create_engine(conn_str)
df.to_sql('PREDICTIONS_TABLE', con=engine, index=False, if_exists='replace')

print("✅ Uploaded to Snowflake!")
