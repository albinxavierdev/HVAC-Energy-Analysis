### HVAC Database Tables

This document summarizes the core tables used by the HVAC system.

All tables live in the default schema. Connection settings are configured via `config.py`.

---

## dynamic_rules
- **Primary Key**: `rule_id` (UUID)
- **Columns**:
  - `rule_id` UUID NOT NULL
  - `equip_id` VARCHAR(50) NOT NULL
  - `associated_equip_id` VARCHAR(50)
  - `target_point_id` VARCHAR(50) NOT NULL
  - `frequency_predicted` INTEGER NOT NULL
  - `description` TEXT

## bacnet_tables
- **Primary Key**: composite (`equip_id`, `point_id`)
- Note: There is an `id` SERIAL column present, but the effective PK is set to the composite key.
- **Columns**:
  - `id` SERIAL
  - `campus_id` VARCHAR(50) NOT NULL
  - `building_id` VARCHAR(50) NOT NULL
  - `floor_id` VARCHAR(50) NOT NULL
  - `project_equipment_id` VARCHAR(50) NOT NULL
  - `equip_id` VARCHAR(50) NOT NULL
  - `associated_equip_id` VARCHAR(50)
  - `point_id` VARCHAR(50) NOT NULL

## dynamic_control
- **Primary Key**: `control_id` (UUID)
- **Foreign Keys**:
  - `rule_id` → `dynamic_rules(rule_id)` ON DELETE CASCADE
- **Columns**:
  - `control_id` UUID NOT NULL
  - `rule_id` UUID NOT NULL
  - `equip_id` VARCHAR(50) NOT NULL
  - `point_id` VARCHAR(50) NOT NULL
  - `predicted_value` NUMERIC
  - `actual_value` NUMERIC
  - `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  - `predicted_for` TIMESTAMP NOT NULL
  - `status` VARCHAR(20) DEFAULT 'pending'
  - `notes` TEXT

---

### Notes
- The schema is created in `hvac_data.py`. Additional alterations (composite primary key on `bacnet_tables` and ensuring `floor_id`) are applied by `alt_tab.py`.
- Sample/dummy data exists in `hvac_dummy_data.sql`. The `insert_data.py` script loads it safely, adding `ON CONFLICT DO NOTHING` to avoid duplicate insert errors.


