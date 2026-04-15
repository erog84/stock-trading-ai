"""Random Forest model for stock signal prediction."""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.models import BaseModel
from src.utils.logger import logger


class RandomForestModel(BaseModel):
    """Random Forest classifier for buy/sell signal prediction."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 10,
        min_samples_leaf: int = 20,
        random_state: int = 42,
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        )
        self.feature_columns: list[str] = []
        self.metrics: dict[str, float] = {}

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Train the Random Forest model.

        Args:
            X: Feature DataFrame (will drop non-numeric and NaN rows)
            y: Target series (binary: 1=up, 0=down)

        Returns:
            Dictionary of training metrics
        """
        # Store feature columns
        self.feature_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        X_clean = X[self.feature_columns].copy()

        # Drop rows with NaN
        mask = X_clean.notna().all(axis=1) & y.notna()
        X_clean = X_clean[mask]
        y_clean = y[mask]

        if len(X_clean) < 100:
            raise ValueError(f"Not enough training data: {len(X_clean)} rows (need 100+)")

        logger.info(f"Training Random Forest on {len(X_clean)} samples, {len(self.feature_columns)} features")

        self.model.fit(X_clean, y_clean)

        # Training metrics
        y_pred = self.model.predict(X_clean)
        self.metrics = {
            "accuracy": accuracy_score(y_clean, y_pred),
            "precision": precision_score(y_clean, y_pred, zero_division=0),
            "recall": recall_score(y_clean, y_pred, zero_division=0),
            "f1": f1_score(y_clean, y_pred, zero_division=0),
            "n_samples": len(X_clean),
            "n_features": len(self.feature_columns),
        }

        logger.info(f"Training metrics: {self.metrics}")
        return self.metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions (0=down, 1=up)."""
        X_clean = X[self.feature_columns].copy()
        # Fill NaN with column median for prediction
        X_clean = X_clean.fillna(X_clean.median())
        return self.model.predict(X_clean)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate prediction probabilities."""
        X_clean = X[self.feature_columns].copy()
        X_clean = X_clean.fillna(X_clean.median())
        return self.model.predict_proba(X_clean)

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importance scores sorted by importance."""
        importances = dict(zip(self.feature_columns, self.model.feature_importances_))
        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    def save(self, path: str) -> None:
        """Save model to disk."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "feature_columns": self.feature_columns,
            "metrics": self.metrics,
        }, save_path)
        logger.info(f"Model saved to {save_path}")

    def load(self, path: str) -> None:
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_columns = data["feature_columns"]
        self.metrics = data["metrics"]
        logger.info(f"Model loaded from {path}")
