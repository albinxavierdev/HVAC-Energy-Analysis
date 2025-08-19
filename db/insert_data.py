from sqlalchemy import create_engine, text
from pathlib import Path
from config import db_url

def insert_all_from_file():
    engine = create_engine(db_url())

    with engine.begin() as conn:
        sql_path = Path(__file__).resolve().parent / "hvac_dummy_data.sql"
        with open(sql_path, "r") as f:
            sql_data = f.read()

        # Patch inserts: add ON CONFLICT DO NOTHING
        patched_sql = []
        for stmt in sql_data.split(";"):
            stmt = stmt.strip()
            if not stmt:
                continue
            if stmt.lower().startswith("insert into"):
                # Append conflict ignore
                if "dynamic_rules" in stmt:
                    stmt += " ON CONFLICT (rule_id) DO NOTHING"
                elif "bacnet_tables" in stmt:
                    stmt += " ON CONFLICT (id) DO NOTHING"
                elif "dynamic_control" in stmt:
                    stmt += " ON CONFLICT (control_id) DO NOTHING"
            patched_sql.append(stmt)

        # Execute one by one
        for stmt in patched_sql:
            conn.execute(text(stmt))

    print("All data inserted successfully")

if __name__ == "__main__":
    insert_all_from_file()
