import pandas as pd
from sqlalchemy import create_engine

DB_USER = "semantic_user"
DB_PASS = "semantic_pass"
DB_HOST = "postgres-db"
DB_PORT = 5432
DB_NAME = "semanticdb"
CSV_PATH = "/data/Churn_Modelling.csv"

engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

print("===> Lade CSV-Datei ...")
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

table_name = "bank_customers"

with engine.begin() as conn:
    df.head(0).to_sql(table_name, con=conn, if_exists="replace", index=False)
    df.to_sql(table_name, con=conn, if_exists="append", index=False)
    print(f"===> {len(df)} Datensätze in '{table_name}' importiert.")
