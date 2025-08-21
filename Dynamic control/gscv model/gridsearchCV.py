import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib


def _resolve_project_root() -> Path:
    # Ensure repository root is on sys.path so we can import db.config
    current = Path(__file__).resolve()
    project_root = current.parent.parent  # "Dynamic control" directory
    repo_root = project_root.parent  # repository root containing the "db" package
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    # Keep returning project_root for local paths (logs/model files)
    return project_root


def _get_engine():
    _resolve_project_root()
    from db.config import db_url  # noqa: WPS433
    return create_engine(db_url())


def _fetch_dynamic_control(engine) -> pd.DataFrame:
    query = text(
        """
        SELECT
            control_id,
            rule_id,
            equip_id,
            point_id,
            predicted_value,
            actual_value,
            created_at,
            predicted_for,
            status
        FROM dynamic_control
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn)
    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure timestamps are datetime
    for col in ["created_at", "predicted_for"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_convert(None)

    # Target: prefer actual_value; if missing, fall back to predicted_value
    df["target_value"] = df["actual_value"].where(df["actual_value"].notna(), df["predicted_value"]) 

    # Drop rows without a target
    df = df[df["target_value"].notna()].copy()

    # Time-based features
    if "predicted_for" in df.columns:
        df["predicted_for_hour"] = df["predicted_for"].dt.hour
        df["predicted_for_dow"] = df["predicted_for"].dt.dayofweek
    else:
        df["predicted_for_hour"] = None
        df["predicted_for_dow"] = None

    # Time delta (seconds) between created_at and predicted_for (can be negative)
    if set(["created_at", "predicted_for"]).issubset(df.columns):
        delta = (df["created_at"] - df["predicted_for"]).dt.total_seconds()
        df["delta_seconds"] = delta
    else:
        df["delta_seconds"] = None

    return df


def _build_pipeline(numeric_features, categorical_features) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            # No scaler needed for RandomForest
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestRegressor(random_state=42, n_jobs=-1)

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    return pipeline


def _log_run(log_path: Path, content: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(content)


def main() -> None:
    project_root = _resolve_project_root()
    logs_path = project_root / "gscv model" / "random_forest.txt"
    model_path = project_root / "gscv model" / "random_forest_gscv.joblib"

    start_ts = datetime.now().isoformat(timespec="seconds")
    _log_run(logs_path, f"\n=== Random Forest GridSearchCV Run @ {start_ts} ===\n")

    # 1) Load data
    engine = _get_engine()
    df_raw = _fetch_dynamic_control(engine)
    _log_run(logs_path, f"Loaded rows: {len(df_raw)}\n")

    # 2) Feature engineering
    df = _engineer_features(df_raw)
    _log_run(logs_path, f"Rows after target filtering: {len(df)}\n")

    # 3) Select features/target
    feature_columns_numeric = [
        "predicted_value",
        "delta_seconds",
        "predicted_for_hour",
        "predicted_for_dow",
    ]
    feature_columns_categorical = [
        "equip_id",
        "point_id",
        "status",
    ]

    # Keep only relevant columns to avoid accidental leaks
    keep_cols = feature_columns_numeric + feature_columns_categorical + ["target_value"]
    df_model = df[keep_cols].copy()

    X = df_model[feature_columns_numeric + feature_columns_categorical]
    y = df_model["target_value"]

    if X.empty or y.empty:
        _log_run(logs_path, "Insufficient data to train the model.\n")
        print("Insufficient data to train the model.")
        return

    # 4) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    _log_run(
        logs_path,
        f"Train size: {len(X_train)}, Test size: {len(X_test)}\n",
    )

    # 5) Build pipeline
    pipeline = _build_pipeline(feature_columns_numeric, feature_columns_categorical)

    # 6) Define GridSearchCV
    param_grid = {
        "model__n_estimators": [100, 300],
        "model__max_depth": [None, 20],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": ["sqrt", 1.0],
    }

    # Log grid size
    num_candidates = 1
    for values in param_grid.values():
        num_candidates *= len(values)
    _log_run(logs_path, f"Grid candidates: {num_candidates}\n")

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="r2",
        cv=3,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )

    # 7) Fit grid search
    grid.fit(X_train, y_train)
    best_params = grid.best_params_
    best_cv_score = grid.best_score_
    _log_run(logs_path, f"Best CV R2: {best_cv_score:.4f}\n")
    _log_run(logs_path, f"Best Params: {best_params}\n")

    # 8) Evaluate on test set
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    metrics_log = (
        f"Test R2: {r2:.4f}\n"
        f"Test MAE: {mae:.4f}\n"
        f"Test RMSE: {rmse:.4f}\n"
    )
    _log_run(logs_path, metrics_log)

    # 9) Save best model
    joblib.dump(best_model, model_path)
    _log_run(logs_path, f"Best model saved to: {model_path}\n")

    print(
        f"Best CV R2: {best_cv_score:.4f}\n"
        f"Test R2: {r2:.4f}\n"
        f"Test MAE: {mae:.4f}\n"
        f"Test RMSE: {rmse:.4f}"
    )


if __name__ == "__main__":
    main()









