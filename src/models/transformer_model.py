"""Transformer model for time series stock prediction."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd

from src.models.base_dl import BaseDeepLearningModel, HAS_TORCH
from src.utils.logger import logger

if HAS_TORCH:
    import torch
    import torch.nn as nn

    class _PositionalEncoding(nn.Module):
        """Positional encoding for transformer input."""

        def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(dropout)

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

            pe[:, 0::2] = torch.sin(position * div_term)
            if d_model > 1:
                pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])

            pe = pe.unsqueeze(0)
            self.register_buffer("pe", pe)

        def forward(self, x):
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)

    class _TransformerNetwork(nn.Module):
        """Transformer encoder network for time series."""

        def __init__(
            self,
            n_features: int,
            d_model: int = 64,
            n_heads: int = 4,
            n_layers: int = 2,
            d_ff: int = 128,
            dropout: float = 0.2,
        ):
            super().__init__()

            self.input_proj = nn.Linear(n_features, d_model)
            self.pos_encoding = _PositionalEncoding(d_model, dropout=dropout)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

            self.fc = nn.Sequential(
                nn.Linear(d_model, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            x = self.input_proj(x)
            x = self.pos_encoding(x)
            encoded = self.encoder(x)
            pooled = encoded.mean(dim=1)
            return self.fc(pooled)


class TransformerModel(BaseDeepLearningModel):
    """Transformer model for stock direction prediction."""

    def __init__(
        self,
        seq_length: int = 20,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.2,
        batch_size: int = 64,
        epochs: int = 50,
        learning_rate: float = 0.0005,
        patience: int = 10,
    ):
        super().__init__(
            seq_length=seq_length,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            patience=patience,
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout

    def _build_model(self, n_features: int) -> nn.Module:
        return _TransformerNetwork(
            n_features=n_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
        )

    def get_feature_importance(self) -> dict[str, float]:
        if not self.feature_columns:
            return {}
        n = len(self.feature_columns)
        return {col: 1.0 / n for col in self.feature_columns}

    def _get_model_config(self) -> dict:
        return {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
        }

    def _apply_model_config(self, config: dict) -> None:
        self.d_model = config.get("d_model", 64)
        self.n_heads = config.get("n_heads", 4)
        self.n_layers = config.get("n_layers", 2)
        self.d_ff = config.get("d_ff", 128)
        self.dropout = config.get("dropout", 0.2)
