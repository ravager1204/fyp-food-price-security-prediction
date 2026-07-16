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
    f"snowflake://{os.environ['SNOWFLAKE_USER']}:{os.environ['SNOWFLAKE_PASSWORD']}@/"
    f"?account={os.environ['SNOWFLAKE_ACCOUNT']}"
    "&warehouse=COMPUTE_WH&database=FYP_DB&schema=PUBLIC&role=ANALYST"  # not ACCOUNTADMIN
)


# Connect and upload
engine = create_engine(conn_str)
df.to_sql('PREDICTIONS_TABLE', con=engine, index=False, if_exists='replace')

print("✅ Uploaded to Snowflake!")
