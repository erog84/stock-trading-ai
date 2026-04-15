"""Ensemble model that combines predictions from multiple models."""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import joblib

from src.models import BaseModel
from src.utils.logger import logger


class EnsembleModel(BaseModel):
    """Combines predictions from multiple models via weighted averaging."""

    def __init__(self, models: Optional[list[BaseModel]] = None, weights: Optional[list[float]] = None):
        self.models = models or []
        self.weights = weights
        self.metrics: dict[str, float] = {}
        self.feature_columns: list[str] = []

    def add_model(self, model: BaseModel, weight: float = 1.0) -> None:
        """Add a model to the ensemble."""
        self.models.append(model)
        if self.weights is None:
            self.weights = [1.0] * len(self.models)
        else:
            self.weights.append(weight)

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Train all models in the ensemble."""
        if not self.models:
            raise ValueError("No models in ensemble")

        all_metrics = {}
        for i, model in enumerate(self.models):
            name = model.__class__.__name__
            try:
                metrics = model.train(X, y)
                all_metrics[f"{name}_{i}"] = metrics
                logger.info(f"Ensemble member {name}: accuracy={metrics.get('accuracy', 'N/A')}")
            except Exception as e:
                logger.warning(f"Ensemble member {name} failed to train: {e}")

        # Aggregate metrics
        accuracies = [m.get("accuracy", 0) for m in all_metrics.values()]
        self.metrics = {
            "ensemble_accuracy": np.mean(accuracies) if accuracies else 0,
            "n_models": len(self.models),
            "member_metrics": all_metrics,
        }

        # Collect all feature columns
        all_cols = set()
        for model in self.models:
            if hasattr(model, "feature_columns"):
                all_cols.update(model.feature_columns)
        self.feature_columns = list(all_cols)

        return self.metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ensemble predictions (majority vote)."""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate weighted average prediction probabilities."""
        if not self.models:
            raise ValueError("No models in ensemble")

        weights = self.weights or [1.0] * len(self.models)
        weight_sum = sum(weights)
        weights_normalized = [w / weight_sum for w in weights]

        all_probas = []
        valid_weights = []

        for model, weight in zip(self.models, weights_normalized):
            try:
                proba = model.predict_proba(X)
                all_probas.append(proba * weight)
                valid_weights.append(weight)
            except Exception as e:
                logger.warning(f"Ensemble prediction failed for {model.__class__.__name__}: {e}")

        if not all_probas:
            raise RuntimeError("All ensemble models failed to predict")

        # Renormalize if some models failed
        total_weight = sum(valid_weights)
        result = sum(all_probas) / total_weight
        return result

    def get_feature_importance(self) -> dict[str, float]:
        """Average feature importance across all models."""
        all_importances: dict[str, list[float]] = {}

        for model in self.models:
            try:
                imp = model.get_feature_importance()
                for feat, score in imp.items():
                    all_importances.setdefault(feat, []).append(score)
            except Exception:
                pass

        averaged = {feat: np.mean(scores) for feat, scores in all_importances.items()}
        return dict(sorted(averaged.items(), key=lambda x: x[1], reverse=True))

    def save(self, path: str) -> None:
        """Save ensemble metadata (individual models saved separately)."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save individual models
        model_paths = []
        for i, model in enumerate(self.models):
            model_path = str(save_path.parent / f"ensemble_member_{i}.joblib")
            model.save(model_path)
            model_paths.append(model_path)

        joblib.dump({
            "model_paths": model_paths,
            "model_classes": [m.__class__.__name__ for m in self.models],
            "weights": self.weights,
            "metrics": self.metrics,
            "feature_columns": self.feature_columns,
        }, save_path)
        logger.info(f"Ensemble saved to {save_path}")

    def load(self, path: str) -> None:
        """Load ensemble from disk."""
        data = joblib.load(path)
        self.weights = data["weights"]
        self.metrics = data["metrics"]
        self.feature_columns = data["feature_columns"]

        # Load individual models
        from src.models.random_forest import RandomForestModel
        from src.models.xgboost_model import XGBoostModel

        model_map = {
            "RandomForestModel": RandomForestModel,
            "XGBoostModel": XGBoostModel,
        }

        # Try importing DL models
        try:
            from src.models.lstm_model import LSTMModel
            from src.models.transformer_model import TransformerModel
            model_map["LSTMModel"] = LSTMModel
            model_map["TransformerModel"] = TransformerModel
        except (ImportError, NameError):
            pass

        self.models = []
        for model_path, model_class in zip(data["model_paths"], data["model_classes"]):
            cls = model_map.get(model_class)
            if cls is None:
                logger.warning(f"Unknown model class: {model_class}")
                continue
            model = cls()
            model.load(model_path)
            self.models.append(model)

        logger.info(f"Ensemble loaded: {len(self.models)} models")
