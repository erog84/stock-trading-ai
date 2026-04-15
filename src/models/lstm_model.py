"""LSTM model for time series stock prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base_dl import BaseDeepLearningModel, HAS_TORCH
from src.utils.logger import logger

if HAS_TORCH:
    import torch
    import torch.nn as nn

    class _LSTMNetwork(nn.Module):
        """LSTM neural network architecture."""

        def __init__(self, n_features: int, hidden_size: int = 128,
                     num_layers: int = 2, dropout: float = 0.3):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            last_hidden = lstm_out[:, -1, :]
            out = self.dropout(last_hidden)
            return self.fc(out)


class LSTMModel(BaseDeepLearningModel):
    """LSTM model for stock direction prediction."""

    def __init__(
        self,
        seq_length: int = 20,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        batch_size: int = 64,
        epochs: int = 50,
        learning_rate: float = 0.001,
        patience: int = 10,
    ):
        super().__init__(
            seq_length=seq_length,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            patience=patience,
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

    def _build_model(self, n_features: int) -> nn.Module:
        return _LSTMNetwork(
            n_features=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

    def get_feature_importance(self) -> dict[str, float]:
        if not self.feature_columns:
            return {}
        n = len(self.feature_columns)
        return {col: 1.0 / n for col in self.feature_columns}

    def _get_model_config(self) -> dict:
        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }

    def _apply_model_config(self, config: dict) -> None:
        self.hidden_size = config.get("hidden_size", 128)
        self.num_layers = config.get("num_layers", 2)
        self.dropout = config.get("dropout", 0.3)
