"""Base class for deep learning models (PyTorch)."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Optional, TYPE_CHECKING
import numpy as np
import pandas as pd

from src.models import BaseModel
from src.models.data_utils import FeatureScaler, create_sequences
from src.utils.logger import logger

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class BaseDeepLearningModel(BaseModel):
    """Base class for PyTorch-based trading models."""

    def __init__(self, seq_length: int = 20, batch_size: int = 64,
                 epochs: int = 50, learning_rate: float = 0.001,
                 patience: int = 10):
        if not HAS_TORCH:
            raise ImportError("PyTorch not installed. Run: pip install torch")

        self.seq_length = seq_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience  # Early stopping patience

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self.scaler = FeatureScaler()
        self.feature_columns: list[str] = []
        self.metrics: dict[str, float] = {}
        self.training_history: list[dict] = []

    @abstractmethod
    def _build_model(self, n_features: int) -> nn.Module:
        """Build the neural network architecture."""
        ...

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Train the deep learning model."""
        # Select numeric features
        self.feature_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        exclude = ["target_return_1d", "target_direction"]
        self.feature_columns = [c for c in self.feature_columns if c not in exclude]

        X_numeric = X[self.feature_columns].copy()

        # Clean NaN
        mask = X_numeric.notna().all(axis=1) & y.notna()
        X_clean = X_numeric[mask]
        y_clean = y[mask].values.astype(np.float32)

        if len(X_clean) < self.seq_length + 50:
            raise ValueError(f"Not enough data: {len(X_clean)} rows (need {self.seq_length + 50}+)")

        # Scale features
        X_scaled = self.scaler.fit_transform(X_clean).astype(np.float32)

        # Create sequences
        X_seq, y_seq = create_sequences(X_scaled, y_clean, self.seq_length)

        # Split (time-series, no shuffle)
        split = int(len(X_seq) * 0.8)
        X_train, X_val = X_seq[:split], X_seq[split:]
        y_train, y_val = y_seq[:split], y_seq[split:]

        # Create DataLoaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train).to(self.device),
            torch.FloatTensor(y_train).to(self.device),
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val).to(self.device),
            torch.FloatTensor(y_val).to(self.device),
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Build model
        n_features = X_seq.shape[2]
        self.model = self._build_model(n_features).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        # Training loop with early stopping
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        logger.info(f"Training {self.__class__.__name__} on {len(X_train)} sequences ({n_features} features)")

        for epoch in range(self.epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = self.model(X_batch).squeeze()
                loss = criterion(output, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # Validate
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    output = self.model(X_batch).squeeze()
                    val_loss += criterion(output, y_batch).item()
            val_loss /= len(val_loader)

            self.training_history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            })

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Compute metrics on validation set
        self.model.eval()
        with torch.no_grad():
            all_preds = []
            all_labels = []
            for X_batch, y_batch in val_loader:
                output = torch.sigmoid(self.model(X_batch).squeeze())
                all_preds.append(output.cpu().numpy())
                all_labels.append(y_batch.cpu().numpy())

            preds = np.concatenate(all_preds)
            labels = np.concatenate(all_labels)
            pred_classes = (preds > 0.5).astype(int)

            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            self.metrics = {
                "accuracy": accuracy_score(labels, pred_classes),
                "precision": precision_score(labels, pred_classes, zero_division=0),
                "recall": recall_score(labels, pred_classes, zero_division=0),
                "f1": f1_score(labels, pred_classes, zero_division=0),
                "val_loss": best_val_loss,
                "epochs_trained": len(self.training_history),
                "n_samples": len(X_train),
                "n_features": n_features,
            }

        logger.info(f"Training complete: {self.metrics}")
        return self.metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate prediction probabilities."""
        if self.model is None:
            raise RuntimeError("Model not trained")

        X_numeric = X[self.feature_columns].copy()
        X_numeric = X_numeric.fillna(X_numeric.median())
        X_scaled = self.scaler.transform(X_numeric).astype(np.float32)

        # If not enough data for a full sequence, pad with the first row
        if len(X_scaled) < self.seq_length:
            padding = np.tile(X_scaled[0], (self.seq_length - len(X_scaled), 1))
            X_scaled = np.vstack([padding, X_scaled])

        # Create sequences (one per row at end)
        X_seq, _ = create_sequences(
            X_scaled,
            np.zeros(len(X_scaled)),
            self.seq_length,
        )

        if len(X_seq) == 0:
            # Single prediction from padded sequence
            X_seq = X_scaled[-self.seq_length:].reshape(1, self.seq_length, -1)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            output = torch.sigmoid(self.model(X_tensor).squeeze(-1))
            proba_up = output.cpu().numpy()

        # Ensure 1D
        if proba_up.ndim == 0:
            proba_up = np.array([proba_up.item()])

        # Return (n, 2) array: [prob_down, prob_up]
        proba_down = 1 - proba_up
        return np.column_stack([proba_down, proba_up])

    def save(self, path: str) -> None:
        """Save model, scaler, and metadata."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        import joblib
        joblib.dump({
            "model_state": self.model.state_dict() if self.model else None,
            "model_class": self.__class__.__name__,
            "feature_columns": self.feature_columns,
            "metrics": self.metrics,
            "seq_length": self.seq_length,
            "training_history": self.training_history,
            "scaler": self.scaler,
            "model_config": self._get_model_config(),
        }, save_path)
        logger.info(f"Model saved to {save_path}")

    def load(self, path: str) -> None:
        """Load model from disk."""
        import joblib
        data = joblib.load(path)

        self.feature_columns = data["feature_columns"]
        self.metrics = data["metrics"]
        self.seq_length = data["seq_length"]
        self.training_history = data.get("training_history", [])
        self.scaler = data["scaler"]

        # Rebuild model architecture and load weights
        config = data.get("model_config", {})
        self._apply_model_config(config)
        n_features = len(self.feature_columns)
        self.model = self._build_model(n_features).to(self.device)

        if data["model_state"] is not None:
            self.model.load_state_dict(data["model_state"])

        logger.info(f"Model loaded from {path}")

    def _get_model_config(self) -> dict:
        """Get model-specific config for serialization. Override in subclasses."""
        return {}

    def _apply_model_config(self, config: dict) -> None:
        """Apply model-specific config from deserialization. Override in subclasses."""
        pass
