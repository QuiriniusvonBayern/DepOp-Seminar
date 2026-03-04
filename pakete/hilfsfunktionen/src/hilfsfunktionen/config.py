import os
from dotenv import load_dotenv

def load_postgres_config():
    """
    Load PostgreSQL connection configuration from environment variables.

    Returns:
        dict: Dictionary containing database connection parameters with keys:
            - host: Database server host
            - port: Database server port
            - dbname: Name of the database
            - user: Username for authentication
            - password: Password for authentication
    """
    load_dotenv()

    host = os.environ["POSTGRES_HOST"]
    port = int(os.environ.get("POSTGRES_PORT", "5432"))
    dbname = os.environ["POSTGRES_DB"]
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]

    return {
        "host": host,
        "port": port,
        "dbname": dbname,
        "user": user,
        "password": password
    }
