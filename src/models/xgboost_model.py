"""XGBoost model for stock signal prediction."""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.models import BaseModel
from src.utils.logger import logger


class XGBoostModel(BaseModel):
    """XGBoost classifier for buy/sell signal prediction."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="logloss",
            use_label_encoder=False,
        )
        self.feature_columns: list[str] = []
        self.metrics: dict[str, float] = {}

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Train the XGBoost model with early stopping support."""
        self.feature_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        X_clean = X[self.feature_columns].copy()

        mask = X_clean.notna().all(axis=1) & y.notna()
        X_clean = X_clean[mask]
        y_clean = y[mask]

        if len(X_clean) < 100:
            raise ValueError(f"Not enough training data: {len(X_clean)} rows")

        logger.info(f"Training XGBoost on {len(X_clean)} samples, {len(self.feature_columns)} features")

        # Split for early stopping
        split_idx = int(len(X_clean) * 0.8)
        X_train, X_val = X_clean.iloc[:split_idx], X_clean.iloc[split_idx:]
        y_train, y_val = y_clean.iloc[:split_idx], y_clean.iloc[split_idx:]

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Metrics on validation set
        y_pred = self.model.predict(X_val)
        self.metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1": f1_score(y_val, y_pred, zero_division=0),
            "n_samples_train": len(X_train),
            "n_samples_val": len(X_val),
            "n_features": len(self.feature_columns),
            "best_iteration": self.model.best_iteration if hasattr(self.model, "best_iteration") else self.model.n_estimators,
        }

        logger.info(f"Validation metrics: {self.metrics}")
        return self.metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_clean = X[self.feature_columns].copy()
        X_clean = X_clean.fillna(X_clean.median())
        return self.model.predict(X_clean)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_clean = X[self.feature_columns].copy()
        X_clean = X_clean.fillna(X_clean.median())
        return self.model.predict_proba(X_clean)

    def get_feature_importance(self) -> dict[str, float]:
        importances = dict(zip(self.feature_columns, self.model.feature_importances_))
        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    def save(self, path: str) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "feature_columns": self.feature_columns,
            "metrics": self.metrics,
        }, save_path)
        logger.info(f"XGBoost model saved to {save_path}")

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_columns = data["feature_columns"]
        self.metrics = data["metrics"]
        logger.info(f"XGBoost model loaded from {path}")
