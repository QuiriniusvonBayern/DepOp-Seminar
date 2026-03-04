from sqlalchemy import create_engine, text
import pandas as pd
from sqlalchemy.engine.url import URL
from .config import load_postgres_config

cfg = load_postgres_config()

def get_engine(
    user: str = cfg["user"],
    password: str = cfg["password"],
    host: str = cfg["host"],
    port: int = cfg["port"],
    dbname: str = cfg["dbname"],
    pool_size: int = 5,
    max_overflow: int = 10,
    pool_timeout: int = 30
):
    """Erzeugt eine SQLAlchemy Engine für PostgreSQL."""
    url = URL.create(
        drivername="postgresql+psycopg2",
        username=user,
        password=password,
        host=host,
        port=port,
        database=dbname
    )

    engine = create_engine(
        url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=pool_timeout,
        future=True
    )
    return engine

def read_table_to_df(sql: str, engine):
    with engine.connect() as conn:
        return pd.read_sql(sql, conn)

def check_engine_connection(engine) -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        print(f"Verbindungsfehler: {e}")
        return False
