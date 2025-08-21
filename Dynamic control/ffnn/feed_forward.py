import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# FFNN Model for HVAC Prediction
class HVAC_FFNN(nn.Module):
    def __init__(self, input_size, hidden1=64, hidden2=32, output_size=1):
        super(HVAC_FFNN, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden1)   # Input → Hidden1
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(hidden1, hidden2)      # Hidden1 → Hidden2
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(hidden2, output_size)  # Hidden2 → Output
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)   # final output (logits)
        return x

# ----------------------------
# Data utilities (reuse RF features)
# ----------------------------

def _resolve_project_root() -> Path:
    current = Path(__file__).resolve()
    project_root = current.parent.parent  # "Dynamic control" directory
    repo_root = project_root.parent  # repository root containing the "db" package
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return project_root


def _get_engine():
    _resolve_project_root()
    from db.config import db_url  
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


def _build_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    # We wrap with an outer ColumnTransformer to keep the same interface as RF
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return preprocessor


def _prepare_tensors(df: pd.DataFrame):
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

    keep_cols = feature_columns_numeric + feature_columns_categorical + ["target_value"]
    df_model = df[keep_cols].copy()

    X = df_model[feature_columns_numeric + feature_columns_categorical]
    y = df_model["target_value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), feature_columns_numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), feature_columns_categorical),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    X_train_np = preprocessor.fit_transform(X_train)
    X_test_np = preprocessor.transform(X_test)

    input_size = X_train_np.shape[1]

    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    # Derive feature counts for logging
    cat_encoder: OneHotEncoder = preprocessor.named_transformers_["cat"]
    cat_sizes = [len(cats) for cats in cat_encoder.categories_]
    numeric_count = len(feature_columns_numeric)
    categorical_count = sum(cat_sizes)

    feature_info = {
        "numeric_feature_count": numeric_count,
        "categorical_cardinalities": dict(zip(feature_columns_categorical, cat_sizes)),
        "categorical_feature_count_after_onehot": categorical_count,
        "total_input_size": input_size,
    }

    return (
        X_train_tensor,
        y_train_tensor,
        X_test_tensor,
        y_test_tensor,
        input_size,
        feature_info,
    )


def train_ffnn_local(hidden1=64, hidden2=32, lr=0.001, epochs=20, batch_size=64):
    engine = _get_engine()
    df_raw = _fetch_dynamic_control(engine)
    df = _engineer_features(df_raw)

    (
        X_train_t,
        y_train_t,
        X_test_t,
        y_test_t,
        input_size,
        feature_info,
    ) = _prepare_tensors(df)

    model = HVAC_FFNN(input_size=input_size, hidden1=hidden1, hidden2=hidden2, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_dataset)
        print(f"Epoch {epoch + 1}/{epochs} - Train MSE: {epoch_loss:.6f}")

    # Evaluation (MSE)
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t)
        test_mse = criterion(test_preds, y_test_t).item()
    print(f"Test MSE: {test_mse:.6f}")

    # Log feature info and model summary
    print("Feature info:", feature_info)
    return model, feature_info, {"test_mse": test_mse}


if __name__ == "__main__":
    _resolve_project_root()
    start_ts = datetime.now().isoformat(timespec="seconds")
    print(f"=== FFNN Training Run @ {start_ts} ===")
    model, feature_info, metrics = train_ffnn_local()
    print(model)  



    