import os
from dotenv import load_dotenv

def load_postgres_config():
    """Lädt die PostgreSQL-Konfiguration aus Umgebungsvariablen."""
    load_dotenv()

    host = os.environ["POSTGRES_HOST"]
    port = int(os.environ.get("POSTGRES_PORT", "5432"))
    db   = os.environ["POSTGRES_DB"]
    user = os.environ["POSTGRES_USER"]
    pwd  = os.environ["POSTGRES_PASSWORD"]

    return {
        "host": host,
        "port": port,
        "dbname": db,
        "user": user,
        "password": pwd
    }
