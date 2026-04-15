"""Data utilities for deep learning models.

Handles sequence creation, feature scaling, and PyTorch dataset management.
"""

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path


class FeatureScaler:
    """Wraps StandardScaler with save/load support."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit scaler and transform data."""
        result = self.scaler.fit_transform(X.values)
        self.fitted = True
        return result

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted scaler."""
        if not self.fitted:
            raise RuntimeError("Scaler not fitted. Call fit_transform first.")
        return self.scaler.transform(X.values)

    def save(self, path: str) -> None:
        joblib.dump(self.scaler, path)

    def load(self, path: str) -> None:
        self.scaler = joblib.load(path)
        self.fitted = True


def create_sequences(
    data: np.ndarray,
    targets: np.ndarray,
    seq_length: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding window sequences for time series models.

    Args:
        data: 2D array of shape (n_samples, n_features)
        targets: 1D array of shape (n_samples,)
        seq_length: Number of time steps per sequence

    Returns:
        X: 3D array of shape (n_sequences, seq_length, n_features)
        y: 1D array of shape (n_sequences,)
    """
    X, y = [], []

    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(targets[i + seq_length])

    return np.array(X), np.array(y)


def prepare_dl_data(
    features: pd.DataFrame,
    feature_columns: list[str],
    target_col: str = "target_direction",
    seq_length: int = 20,
    test_size: float = 0.2,
) -> dict:
    """Prepare data for deep learning models.

    Returns dict with train/test splits as numpy arrays.
    """
    # Clean data
    mask = features[feature_columns + [target_col]].notna().all(axis=1)
    clean = features[mask].copy()

    # Scale features
    scaler = FeatureScaler()
    X_scaled = scaler.fit_transform(clean[feature_columns])
    y = clean[target_col].values

    # Create sequences
    X_seq, y_seq = create_sequences(X_scaled, y, seq_length)

    # Time-series split
    split_idx = int(len(X_seq) * (1 - test_size))

    return {
        "X_train": X_seq[:split_idx],
        "y_train": y_seq[:split_idx],
        "X_test": X_seq[split_idx:],
        "y_test": y_seq[split_idx:],
        "scaler": scaler,
        "feature_columns": feature_columns,
        "seq_length": seq_length,
    }
