from sqlalchemy import create_engine, inspect, text, MetaData, Table
from sqlalchemy.engine import reflection

from config import db_url

# =========================================================
# 1. Database connection settings
# =========================================================
# Moved to config.py; override via environment variables if needed.

# =========================================================
# 2. Connect to DB
# =========================================================
engine = create_engine(db_url())
inspector = inspect(engine)
metadata = MetaData()

# =========================================================
# 3. Print all data from all tables
# =========================================================
def print_all_data():
    print("\n=== ALL TABLE DATA ===\n")
    with engine.connect() as conn:
        for table_name in inspector.get_table_names():
            print(f"\n--- {table_name.upper()} ---")
            result = conn.execute(text(f"SELECT * FROM {table_name} LIMIT 10"))  # limit for safety
            rows = result.fetchall()
            if rows:
                for row in rows:
                    print(row)
            else:
                print("(no data)")

# =========================================================
# 4. Print schema structure
# =========================================================
def print_schema_structure():
    print("\n=== SCHEMA STRUCTURE ===\n")
    for table_name in inspector.get_table_names():
        print(f"\n--- {table_name.upper()} ---")
        columns = inspector.get_columns(table_name)
        for col in columns:
            col_details = f"{col['name']} {col['type']}"
            if col.get("nullable") is False:
                col_details += " NOT NULL"
            if col.get("default") is not None:
                col_details += f" DEFAULT {col['default']}"
            print("   -", col_details)

# =========================================================
# 5. Print relationships (PK & FK)
# =========================================================
def print_relationships():
    print("\n=== RELATIONSHIPS ===\n")
    for table_name in inspector.get_table_names():
        print(f"\n--- {table_name.upper()} ---")

        # Primary keys
        pk = inspector.get_pk_constraint(table_name)
        if pk and pk.get("constrained_columns"):
            print(f"   Primary Key: {pk['constrained_columns']}")

        # Foreign keys
        fks = inspector.get_foreign_keys(table_name)
        if fks:
            for fk in fks:
                print(f"   Foreign Key: {fk['constrained_columns']} → {fk['referred_table']}({fk['referred_columns']})")

# =========================================================
# 6. Run all
# =========================================================
if __name__ == "__main__":
    print_all_data()
    print_schema_structure()
    print_relationships()
