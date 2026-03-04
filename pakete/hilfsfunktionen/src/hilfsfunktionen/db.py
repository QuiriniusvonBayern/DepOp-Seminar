from sqlalchemy import create_engine, text
import pandas as pd
from sqlalchemy.engine.url import URL
from .config import load_postgres_config

config = load_postgres_config()


def get_engine(
    user: str = config["user"],
    password: str = config["password"],
    host: str = config["host"],
    port: int = config["port"],
    database: str = config["dbname"],
    pool_size: int = 5,
    max_overflow: int = 10,
    pool_timeout: int = 30,
):
    """
    Create a SQLAlchemy engine for PostgreSQL database connection.

    Args:
        user: Database username
        password: Database password
        host: Database host address
        port: Database port
        database: Database name
        pool_size: Number of connections to maintain in pool
        max_overflow: Maximum overflow connections beyond pool_size
        pool_timeout: Seconds to wait before timing out on connection acquisition

    Returns:
        SQLAlchemy engine instance configured for PostgreSQL with connection pooling
    """
    connection_url = URL.create(
        drivername="postgresql+psycopg2",
        username=user,
        password=password,
        host=host,
        port=port,
        database=database,
    )

    engine = create_engine(
        connection_url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=pool_timeout,
        future=True,
    )
    return engine


def read_table_to_dataframe(sql_query: str, engine):
    """
    Execute SQL query and return results as pandas DataFrame.

    Args:
        sql_query: SQL query string to execute
        engine: SQLAlchemy engine for database connection

    Returns:
        DataFrame containing query results
    """
    with engine.connect() as connection:
        return pd.read_sql(sql_query, connection)


def verify_database_connection(engine) -> bool:
    """
    Test database connection by executing a simple query.

    Args:
        engine: SQLAlchemy engine to test

    Returns:
        True if connection successful, False otherwise
    """
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception as error:
        print(f"Database connection error: {error}")
        return False
