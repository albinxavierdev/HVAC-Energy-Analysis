# Project Updates

This file records key changes made to the HVAC project.

## 2025-08-19
- Added centralized DB config `db/config.py` and switched scripts to use it.
- Moved DB-related scripts and data into `db/`:
  - `db/hvac_data.py`, `db/hvac_dummy_data.sql`, `db/insert_data.py`, `db/display_data.py`, `db/config.py`.
- Created documentation:
  - `docs/tables.md` (schema overview)
  - `docs/relationship.txt` (relationships, recommendations)
- Added `.gitignore` and `.gitattributes` for clean commits and normalized line endings.
- Added linear regression pipeline under `Dynamic control/lrmodel/`:
  - `train_linear_model.py` to pull data from Postgres, preprocess, train, evaluate, and log steps/metrics to `linear_model.txt`.
  - Ignored model artifacts (`*.joblib`, etc.) to avoid committing trained models.
