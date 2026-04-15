"""ML models module."""

from abc import ABC, abstractmethod
from typing import Any, Optional
import pandas as pd
import numpy as np


class BaseModel(ABC):
    """Base interface for all trading ML models."""

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Train the model. Returns training metrics."""
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate prediction probabilities. Returns (n_samples, n_classes) array."""
        ...

    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importance scores."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model artifacts to disk."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model artifacts from disk."""
        ...
