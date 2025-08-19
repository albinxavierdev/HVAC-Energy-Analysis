from sqlalchemy import create_engine, text
from config import db_url


SQL_SCRIPT = """
-- =========================================================
-- Drop old tables if they exist (order matters due to FK)
-- =========================================================
DROP TABLE IF EXISTS dynamic_control CASCADE;
DROP TABLE IF EXISTS dynamic_rules CASCADE;
DROP TABLE IF EXISTS bacnet_tables CASCADE;

-- =========================================================
-- Create Tables
-- =========================================================

-- 1. dynamic_rules (UUID PK)
CREATE TABLE IF NOT EXISTS dynamic_rules (
    rule_id UUID PRIMARY KEY,
    equip_id VARCHAR(50) NOT NULL,
    associated_equip_id VARCHAR(50),
    target_point_id VARCHAR(50) NOT NULL,
    frequency_predicted INTEGER NOT NULL, -- in seconds
    description TEXT
);

-- 2. bacnet_tables (SERIAL PK is fine here)
CREATE TABLE IF NOT EXISTS bacnet_tables (
    id SERIAL PRIMARY KEY,
    campus_id VARCHAR(50) NOT NULL,
    building_id VARCHAR(50) NOT NULL,
    floor_id VARCHAR(50) NOT NULL,
    project_equipment_id VARCHAR(50) NOT NULL,
    equip_id VARCHAR(50) NOT NULL,
    associated_equip_id VARCHAR(50),
    point_id VARCHAR(50) NOT NULL
);

-- 3. dynamic_control (UUID PK + UUID FK to dynamic_rules)
CREATE TABLE IF NOT EXISTS dynamic_control (
    control_id UUID PRIMARY KEY,
    rule_id UUID NOT NULL REFERENCES dynamic_rules(rule_id) ON DELETE CASCADE,
    equip_id VARCHAR(50) NOT NULL,
    point_id VARCHAR(50) NOT NULL,
    predicted_value NUMERIC,
    actual_value NUMERIC,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    predicted_for TIMESTAMP NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    notes TEXT
);
"""



# =========================================================
# 4. Main execution
# =========================================================
def main():
    engine = create_engine(db_url())

    with engine.begin() as conn:
        # Create tables
        conn.execute(text(SQL_SCRIPT))


    print("Tables created successfully.")

if __name__ == "__main__":
    main()
