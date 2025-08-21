import os

# Centralized database configuration
# Override via environment variables if needed.

DB_USER = os.getenv("HVAC_DB_USER", "postgres")
DB_PASSWORD = os.getenv("HVAC_DB_PASSWORD", "1234")
DB_HOST = os.getenv("HVAC_DB_HOST", "localhost")
DB_PORT = os.getenv("HVAC_DB_PORT", "5432")
DB_NAME = os.getenv("HVAC_DB_NAME", "hvac_system")


def db_url() -> str:
    return f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


