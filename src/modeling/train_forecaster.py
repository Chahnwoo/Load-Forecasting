#!/usr/bin/env python3

import argparse
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Dummy classes to avoid NameError when torch is not installed
    class Dataset: pass
    class nn:
        class Module: pass
        class MSELoss: pass
        class LSTM: pass
        class Linear: pass
        class Sequential: pass
        class ReLU: pass
        class TransformerEncoderLayer: pass
        class TransformerEncoder: pass
    class torch:
        class optim:
            class Adam: pass
        @staticmethod
        def tensor(*args, **kwargs): pass
        @staticmethod
        def zeros(*args, **kwargs): pass
        @staticmethod
        def arange(*args, **kwargs): pass
        @staticmethod
        def exp(*args, **kwargs): pass
        @staticmethod
        def sin(*args, **kwargs): pass
        @staticmethod
        def cos(*args, **kwargs): pass
        @staticmethod
        def no_grad(*args, **kwargs): pass


# =============================================================================
# Utility
# =============================================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def print_step(message: str):
    print(f"\n[INFO] {message}")


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = safe_mape(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    mean_actual = np.mean(y_true)
    mean_forecast = np.mean(y_pred)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "BIAS": bias,
        "MEAN_ACTUAL": mean_actual,
        "MEAN_FORECAST": mean_forecast,
    }


def pretty_print_metrics(metrics: dict):
    print("\n" + "=" * 80)
    print("VALIDATION METRICS")
    print("=" * 80)
    for key in ["RMSE", "MAE", "MAPE", "BIAS", "MEAN_ACTUAL", "MEAN_FORECAST", "MSE"]:
        value = metrics[key]
        if pd.isna(value):
            formatted = "NaN"
        elif key == "MAPE":
            formatted = f"{value:,.4f}%"
        else:
            formatted = f"{value:,.4f}"
        print(f"{key:<15}: {formatted}")
    print("=" * 80)


def parse_predict_month(predict_month: str):
    """
    Parse YYYY-MM and return:
    - month_start: first timestamp of that month
    - next_month_start: first timestamp of next month
    - train_end: last timestamp before predict month begins
    """
    try:
        year_str, month_str = predict_month.split("-")
        year = int(year_str)
        month = int(month_str)
    except Exception as e:
        raise ValueError(
            f"Invalid predict_month='{predict_month}'. Expected format YYYY-MM."
        ) from e

    if not (1 <= month <= 12):
        raise ValueError(
            f"Invalid predict_month='{predict_month}'. Month must be between 01 and 12."
        )

    month_start = pd.Timestamp(year=year, month=month, day=1, hour=0, minute=0, second=0)

    if month == 12:
        next_month_start = pd.Timestamp(year=year + 1, month=1, day=1, hour=0, minute=0, second=0)
    else:
        next_month_start = pd.Timestamp(year=year, month=month + 1, day=1, hour=0, minute=0, second=0)

    train_end = month_start - pd.Timedelta(seconds=1)
    return month_start, next_month_start, train_end


# =============================================================================
# Data prep
# =============================================================================

REQUIRED_COLUMNS = {
    "region",
    "time_key",
    "time_utc",
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "shortwave_radiation",
    "cdd_65f",
    "hdd_65f",
    "load_mw",
    "is_weekend",
    "US_federal_holidays",
    "state_holidays",
    "load_previous_week",
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["time_utc"] = pd.to_datetime(df["time_utc"], errors="coerce")
    if df["time_utc"].isna().any():
        bad_count = int(df["time_utc"].isna().sum())
        raise ValueError(f"Found {bad_count} rows with invalid time_utc values.")

    df["year"] = df["time_utc"].dt.year
    df["month"] = df["time_utc"].dt.month
    df["day"] = df["time_utc"].dt.day
    df["day_of_week"] = df["time_utc"].dt.dayofweek
    df["hour"] = df["time_utc"].dt.hour
    df["day_of_year"] = df["time_utc"].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    return df


def drop_bad_rows(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    before = len(df)
    missing_target = int(df["load_mw"].isna().sum())
    missing_prev_week = int(df["load_previous_week"].isna().sum())

    df = df.dropna(subset=["load_mw", "load_previous_week"]).copy()

    after = len(df)
    dropped = before - after

    print(
        f"[INFO] {split_name}: dropped {dropped:,} rows "
        f"(missing load_mw={missing_target:,}, missing load_previous_week={missing_prev_week:,})."
    )
    return df


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = [col for col in X.columns if X[col].dtype == "object" or pd.api.types.is_string_dtype(X[col])]
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        sparse_threshold=0.0,
    )


# =============================================================================
# Torch sequence datasets/models
# =============================================================================

class SequenceDataset(Dataset):
    def __init__(self, X_seq: np.ndarray, y_seq: np.ndarray):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.y_seq = torch.tensor(y_seq, dtype=torch.float32)

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.y_seq[idx]


class LSTMRegressorNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        pred = self.head(last_hidden).squeeze(-1)
        return pred


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerRegressorNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        return self.head(x).squeeze(-1)


class BiLSTMRegressorNet(nn.Module):
    """Bidirectional LSTM — Paper 1 (s43067) cites Bi-LSTM as outperforming unidirectional."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class STCALNet(nn.Module):
    """
    Spatio-Temporal CNN-Attention LSTM (ST-CALNet) from Paper 2 (electronics-14-02514).
    Architecture: 1D-CNN → LSTM → Scaled-Dot-Product Attention + Residual+LN → LSTM → Dense.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        cnn_channels: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm1 = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self._attn_scale = hidden_dim ** 0.5
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=max(1, num_layers - 1),
            batch_first=True,
            dropout=dropout if num_layers > 2 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: [B, T, F]
        z = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, T, C]
        h, _ = self.lstm1(z)                                  # [B, T, H]
        Q = self.W_Q(h)
        K = self.W_K(h)
        V = self.W_V(h)
        attn = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) / self._attn_scale, dim=-1)
        context = torch.bmm(attn, V)                          # [B, T, H]
        h_prime = self.layer_norm(h + context)                # residual + LN
        out, _ = self.lstm2(h_prime)
        return self.head(out[:, -1, :]).squeeze(-1)


class TorchSequenceRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        model_type: str = "lstm",
        lookback: int = 24,
        epochs: int = 10,
        batch_size: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        d_model: int = 64,
        nhead: int = 4,
        dim_feedforward: int = 128,
        cnn_channels: int = 64,
        device: str = "auto",
        seed: int = 42,
        verbose: bool = True,
    ):
        self.model_type = model_type
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.cnn_channels = cnn_channels
        self.device = device
        self.seed = seed
        self.verbose = verbose

    def _resolve_device(self):
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _build_sequences(self, X, y, groups):
        X_seq = []
        y_seq = []
        valid_indices = []

        df_idx = pd.DataFrame({
            "group": groups,
            "orig_idx": np.arange(len(X)),
        })

        for group_value in df_idx["group"].unique():
            idxs = df_idx.loc[df_idx["group"] == group_value, "orig_idx"].values
            idxs = np.asarray(idxs)
            if len(idxs) <= self.lookback:
                continue

            for i in range(self.lookback, len(idxs)):
                seq_idx = idxs[i - self.lookback:i]
                target_idx = idxs[i]
                X_seq.append(X[seq_idx])
                y_seq.append(y[target_idx])
                valid_indices.append(target_idx)

        if not X_seq:
            raise ValueError(
                "No sequences were created. Try reducing --lookback or ensure enough rows per region."
            )

        return np.asarray(X_seq), np.asarray(y_seq), np.asarray(valid_indices)

    def fit(self, X, y, groups):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for LSTM/Transformer models. Install torch first."
            )

        set_seed(self.seed)
        self.device_ = self._resolve_device()

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        groups = np.asarray(groups)

        self.y_scaler_ = StandardScaler()
        y_scaled = self.y_scaler_.fit_transform(y.reshape(-1, 1)).reshape(-1)

        X_seq, y_seq, valid_idx = self._build_sequences(X, y_scaled, groups)
        self.train_valid_indices_ = valid_idx

        train_ds = SequenceDataset(X_seq, y_seq)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        input_dim = X_seq.shape[-1]

        if self.model_type == "lstm":
            self.model_ = LSTMRegressorNet(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout,
            )
        elif self.model_type == "transformer":
            self.model_ = TransformerRegressorNet(
                input_dim=input_dim,
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
            )
        elif self.model_type == "bilstm":
            self.model_ = BiLSTMRegressorNet(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout,
            )
        elif self.model_type == "stcalnet":
            self.model_ = STCALNet(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                cnn_channels=self.cnn_channels,
                dropout=self.dropout,
            )
        else:
            raise ValueError(f"Unsupported sequence model_type: {self.model_type}")

        self.model_.to(self.device_)
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.MSELoss()

        epoch_bar = tqdm(range(self.epochs), desc=f"{self.model_type.upper()} training", leave=False)
        for _ in epoch_bar:
            self.model_.train()
            running_loss = 0.0
            n_samples = 0

            for xb, yb in train_loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)

                optimizer.zero_grad()
                preds = self.model_(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()

                batch_size = xb.size(0)
                running_loss += loss.item() * batch_size
                n_samples += batch_size

            avg_loss = running_loss / max(n_samples, 1)
            epoch_bar.set_postfix({"train_mse_loss": f"{avg_loss:.6f}"})

        return self

    def predict(self, X, groups):
        X = np.asarray(X, dtype=np.float32)
        groups = np.asarray(groups)

        dummy_y = np.zeros(len(X), dtype=np.float32)
        X_seq, _, valid_idx = self._build_sequences(X, dummy_y, groups)

        self.model_.eval()
        preds = []

        with torch.no_grad():
            for start in range(0, len(X_seq), self.batch_size):
                xb = torch.tensor(X_seq[start:start + self.batch_size], dtype=torch.float32).to(self.device_)
                out = self.model_(xb).cpu().numpy()
                preds.append(out)

        preds = np.concatenate(preds, axis=0)
        preds = self.y_scaler_.inverse_transform(preds.reshape(-1, 1)).reshape(-1)
        return preds, valid_idx


# =============================================================================
# Model factory
# =============================================================================

def build_model(args):
    if args.model == "linear":
        return LinearRegression()

    if args.model == "ridge":
        return Ridge(alpha=args.ridge_alpha, random_state=args.seed)

    if args.model == "hinge_regression":
        return LinearSVR(
            C=args.svr_c,
            epsilon=args.svr_epsilon,
            max_iter=args.svr_max_iter,
            random_state=args.seed,
        )

    if args.model == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is required for --model xgboost. Install it with: pip install xgboost"
            )

        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=args.xgb_n_estimators,
            learning_rate=args.xgb_learning_rate,
            max_depth=args.xgb_max_depth,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample_bytree,
            reg_alpha=args.xgb_reg_alpha,
            reg_lambda=args.xgb_reg_lambda,
            min_child_weight=args.xgb_min_child_weight,
            tree_method=args.xgb_tree_method,
            random_state=args.seed,
            n_jobs=-1,
        )

    if args.model == "random_forest":
        return RandomForestRegressor(
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth if args.rf_max_depth > 0 else None,
            min_samples_leaf=args.rf_min_samples_leaf,
            n_jobs=-1,
            random_state=args.seed,
        )

    if args.model == "lightgbm":
        if not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "LightGBM is required for --model lightgbm. Install it with: pip install lightgbm"
            )
        return lgb.LGBMRegressor(
            n_estimators=args.lgbm_n_estimators,
            learning_rate=args.lgbm_learning_rate,
            max_depth=args.lgbm_max_depth,
            num_leaves=args.lgbm_num_leaves,
            subsample=args.lgbm_subsample,
            colsample_bytree=args.lgbm_colsample_bytree,
            reg_alpha=args.lgbm_reg_alpha,
            reg_lambda=args.lgbm_reg_lambda,
            random_state=args.seed,
            n_jobs=-1,
            verbose=-1,
        )

    if args.model in {"lstm", "transformer", "bilstm", "stcalnet"}:
        return TorchSequenceRegressor(
            model_type=args.model,
            lookback=args.lookback,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            d_model=args.d_model,
            nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            cnn_channels=args.cnn_channels,
            device=args.device,
            seed=args.seed,
            verbose=True,
        )

    raise ValueError(f"Unsupported model: {args.model}")


# =============================================================================
# Training / evaluation pipeline
# =============================================================================

def prepare_base_data(df: pd.DataFrame, predict_month: str):
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = build_features(df)

    month_start, next_month_start, train_end = parse_predict_month(predict_month)

    train_df = df[df["time_utc"] < month_start].copy()
    valid_df = df[(df["time_utc"] >= month_start) & (df["time_utc"] < next_month_start)].copy()

    if train_df.empty:
        raise ValueError(
            f"No training rows found before predict_month={predict_month}."
        )
    if valid_df.empty:
        raise ValueError(
            f"No validation rows found within predict_month={predict_month}."
        )

    print(f"[INFO] predict_month           : {predict_month}")
    print(f"[INFO] training window end     : {train_end}")
    print(f"[INFO] validation window start : {month_start}")
    print(f"[INFO] validation window end   : {next_month_start - pd.Timedelta(seconds=1)}")

    train_df = drop_bad_rows(train_df, "TRAIN")
    valid_df = drop_bad_rows(valid_df, "VALID")

    if train_df.empty:
        raise ValueError("No usable training rows remain.")
    if valid_df.empty:
        raise ValueError("No usable validation rows remain.")

    return train_df, valid_df, month_start, next_month_start, train_end


def build_feature_matrices(train_df: pd.DataFrame, valid_df: pd.DataFrame):
    """
    Sort once, then build X/y/group arrays from the already-sorted dataframes.
    This avoids pandas index-misalignment issues.
    """
    train_df = train_df.sort_values(["region", "time_utc"]).reset_index(drop=True)
    valid_df = valid_df.sort_values(["region", "time_utc"]).reset_index(drop=True)

    y_train = train_df["load_mw"].values
    y_valid = valid_df["load_mw"].values

    train_groups = train_df["region"].astype(str).values
    valid_groups = valid_df["region"].astype(str).values

    columns_to_drop = ["load_mw", "time_utc", "time_key"]

    X_train_model = train_df.drop(columns=columns_to_drop, errors="ignore").copy()
    X_valid_model = valid_df.drop(columns=columns_to_drop, errors="ignore").copy()

    preprocessor = make_preprocessor(X_train_model)
    X_train_trans = preprocessor.fit_transform(X_train_model)
    X_valid_trans = preprocessor.transform(X_valid_model)

    return {
        "train_df": train_df,
        "valid_df": valid_df,
        "X_train": X_train_trans,
        "X_valid": X_valid_trans,
        "y_train": y_train,
        "y_valid": y_valid,
        "train_groups": train_groups,
        "valid_groups": valid_groups,
        "preprocessor": preprocessor,
    }


def fit_and_predict(args, prepared: dict):
    model = build_model(args)

    X_train = prepared["X_train"]
    X_valid = prepared["X_valid"]
    y_train = prepared["y_train"]
    y_valid = prepared["y_valid"]
    train_df = prepared["train_df"]
    valid_df = prepared["valid_df"]
    train_groups = prepared["train_groups"]
    valid_groups = prepared["valid_groups"]

    if args.model in {"linear", "ridge", "hinge_regression", "xgboost", "random_forest", "lightgbm"}:
        model.fit(X_train, y_train)
        valid_preds = model.predict(X_valid)
        eval_valid_df = valid_df.copy()
        eval_y_valid = y_valid.copy()

    elif args.model in {"lstm", "transformer", "bilstm", "stcalnet"}:
        model.fit(X_train, y_train, groups=train_groups)
        valid_preds, valid_idx = model.predict(X_valid, groups=valid_groups)
        eval_valid_df = valid_df.iloc[valid_idx].reset_index(drop=True)
        eval_y_valid = y_valid[valid_idx]

    else:
        raise ValueError(f"Unsupported model: {args.model}")

    metrics = compute_metrics(eval_y_valid, valid_preds)

    return {
        "model": model,
        "valid_preds": valid_preds,
        "eval_valid_df": eval_valid_df,
        "eval_y_valid": eval_y_valid,
        "metrics": metrics,
        "train_df": train_df,
        "valid_df_full": valid_df,
    }


def save_outputs(args, results: dict, split_info: dict):
    output_path = Path(args.output_predictions)
    output_metrics_path = (
        Path(args.output_metrics)
        if args.output_metrics is not None
        else output_path.with_name(output_path.stem + "_metrics.csv")
    )

    eval_valid_df = results["eval_valid_df"].copy()
    valid_preds = results["valid_preds"]
    metrics = results["metrics"]
    train_df = results["train_df"]

    pred_df = eval_valid_df[["region", "time_key", "time_utc", "load_mw"]].copy()
    pred_df = pred_df.rename(columns={"load_mw": "actual_load_mw"})
    pred_df["predicted_load_mw"] = valid_preds
    pred_df["error"] = pred_df["predicted_load_mw"] - pred_df["actual_load_mw"]
    pred_df["absolute_error"] = np.abs(pred_df["error"])
    pred_df["squared_error"] = pred_df["error"] ** 2

    nonzero_mask = pred_df["actual_load_mw"] != 0
    pred_df["ape_percent"] = np.nan
    pred_df.loc[nonzero_mask, "ape_percent"] = (
        np.abs(pred_df.loc[nonzero_mask, "error"]) /
        np.abs(pred_df.loc[nonzero_mask, "actual_load_mw"])
    ) * 100.0

    pred_df.to_csv(output_path, index=False)

    metrics_df = pd.DataFrame([
        {
            "model": args.model,
            "predict_month": args.predict_month,
            "train_rows_used": len(train_df),
            "validation_rows_scored": len(pred_df),
            "training_end_time": str(split_info["train_end"]),
            "validation_start_time": str(split_info["month_start"]),
            "validation_end_time": str(split_info["next_month_start"] - pd.Timedelta(seconds=1)),
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "MAPE": metrics["MAPE"],
            "BIAS": metrics["BIAS"],
            "MEAN_ACTUAL": metrics["MEAN_ACTUAL"],
            "MEAN_FORECAST": metrics["MEAN_FORECAST"],
            "MSE": metrics["MSE"],
        }
    ])
    metrics_df.to_csv(output_metrics_path, index=False)

    print(f"[INFO] Saved validation predictions to: {output_path.resolve()}")
    print(f"[INFO] Saved validation metrics to: {output_metrics_path.resolve()}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Modular load forecasting pipeline with month-based validation."
    )
    parser.add_argument("csv_path", type=str, help="Path to input CSV.")
    parser.add_argument(
        "--predict_month",
        type=str,
        required=True,
        help="Month to validate on, in YYYY-MM format. Example: 2025-12",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="linear",
        choices=[
            "linear", "ridge", "hinge_regression",
            "xgboost", "random_forest", "lightgbm",
            "lstm", "transformer", "bilstm", "stcalnet",
        ],
        help="Which model backend to use.",
    )
    parser.add_argument(
        "--output_predictions",
        type=str,
        default="validation_predictions.csv",
        help="Path to save validation predictions CSV.",
    )
    parser.add_argument(
        "--output_metrics",
        type=str,
        default=None,
        help="Optional path to save metrics CSV.",
    )

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--ridge_alpha", type=float, default=1.0)

    parser.add_argument("--svr_c", type=float, default=1.0)
    parser.add_argument("--svr_epsilon", type=float, default=0.1)
    parser.add_argument("--svr_max_iter", type=int, default=5000)

    parser.add_argument("--xgb_n_estimators", type=int, default=300)
    parser.add_argument("--xgb_learning_rate", type=float, default=0.05)
    parser.add_argument("--xgb_max_depth", type=int, default=6)
    parser.add_argument("--xgb_subsample", type=float, default=0.8)
    parser.add_argument("--xgb_colsample_bytree", type=float, default=0.8)
    parser.add_argument("--xgb_reg_alpha", type=float, default=0.0)
    parser.add_argument("--xgb_reg_lambda", type=float, default=1.0)
    parser.add_argument("--xgb_min_child_weight", type=float, default=1.0)
    parser.add_argument("--xgb_tree_method", type=str, default="hist")

    # Random Forest
    parser.add_argument("--rf_n_estimators", type=int, default=300)
    parser.add_argument("--rf_max_depth", type=int, default=0, help="0 = unlimited")
    parser.add_argument("--rf_min_samples_leaf", type=int, default=1)

    # LightGBM
    parser.add_argument("--lgbm_n_estimators", type=int, default=300)
    parser.add_argument("--lgbm_learning_rate", type=float, default=0.05)
    parser.add_argument("--lgbm_max_depth", type=int, default=-1, help="-1 = unlimited")
    parser.add_argument("--lgbm_num_leaves", type=int, default=31)
    parser.add_argument("--lgbm_subsample", type=float, default=0.8)
    parser.add_argument("--lgbm_colsample_bytree", type=float, default=0.8)
    parser.add_argument("--lgbm_reg_alpha", type=float, default=0.0)
    parser.add_argument("--lgbm_reg_lambda", type=float, default=1.0)

    parser.add_argument("--lookback", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=128)
    # ST-CALNet CNN channels
    parser.add_argument("--cnn_channels", type=int, default=64)

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    total_steps = 7
    progress = tqdm(total=total_steps, desc="Forecasting pipeline", unit="step")

    print_step("Reading CSV...")
    df = pd.read_csv(csv_path)
    progress.update(1)

    print_step("Preparing train/validation split from predict_month...")
    train_df, valid_df, month_start, next_month_start, train_end = prepare_base_data(
        df, args.predict_month
    )
    progress.update(1)

    print_step("Building shared feature pipeline...")
    prepared = build_feature_matrices(train_df, valid_df)
    progress.update(1)

    print_step(f"Training model: {args.model}")
    results = fit_and_predict(args, prepared)
    progress.update(1)

    print_step("Computing validation metrics...")
    pretty_print_metrics(results["metrics"])
    progress.update(1)

    print_step("Saving predictions and metrics...")
    split_info = {
        "month_start": month_start,
        "next_month_start": next_month_start,
        "train_end": train_end,
    }
    save_outputs(args, results, split_info)
    progress.update(1)

    print_step("Done.")
    progress.update(1)
    progress.close()


if __name__ == "__main__":
    main()