"""Model training pipeline with walk-forward validation."""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

from src.models import BaseModel
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.utils.logger import logger

# Features to exclude from model training
EXCLUDE_COLUMNS = [
    "open", "high", "low", "close", "volume", "ticker",
    "target_return_1d", "target_direction",
]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get columns suitable for model training."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in EXCLUDE_COLUMNS]


class ModelTrainer:
    """Handles model training, evaluation, and walk-forward validation."""

    def __init__(self, model_dir: str = "data/models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def train_model(
        self,
        features: pd.DataFrame,
        model_type: str = "random_forest",
        target_col: str = "target_direction",
        test_size: float = 0.2,
    ) -> tuple[BaseModel, dict]:
        """Train a model on the feature matrix.

        Uses time-series split (not random) to avoid look-ahead bias.
        """
        # Select features
        feature_cols = get_feature_columns(features)
        X = features[feature_cols]
        y = features[target_col]

        # Time-series split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Create model
        model = self._create_model(model_type)

        # Train
        train_metrics = model.train(X_train, y_train)

        # Evaluate on test set
        y_pred = model.predict(X_test)
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        test_metrics = {
            "test_accuracy": accuracy_score(y_test[y_test.notna()], y_pred[y_test.notna()]),
            "test_precision": precision_score(y_test[y_test.notna()], y_pred[y_test.notna()], zero_division=0),
            "test_recall": recall_score(y_test[y_test.notna()], y_pred[y_test.notna()], zero_division=0),
            "test_f1": f1_score(y_test[y_test.notna()], y_pred[y_test.notna()], zero_division=0),
        }

        all_metrics = {**train_metrics, **test_metrics}

        # Save model
        model_path = self.model_dir / f"{model_type}_latest.joblib"
        model.save(str(model_path))

        logger.info(f"Model trained and saved. Test accuracy: {test_metrics['test_accuracy']:.4f}")
        return model, all_metrics

    def walk_forward_validate(
        self,
        features: pd.DataFrame,
        model_type: str = "random_forest",
        target_col: str = "target_direction",
        train_window: int = 252,  # ~1 year of trading days
        test_window: int = 21,    # ~1 month
    ) -> pd.DataFrame:
        """Walk-forward validation to simulate real trading conditions.

        Trains on a rolling window, predicts the next period, then rolls forward.
        """
        feature_cols = get_feature_columns(features)
        results = []

        n_splits = (len(features) - train_window) // test_window
        logger.info(f"Walk-forward validation: {n_splits} splits, train={train_window}, test={test_window}")

        for i in range(n_splits):
            train_start = i * test_window
            train_end = train_start + train_window
            test_end = min(train_end + test_window, len(features))

            X_train = features.iloc[train_start:train_end][feature_cols]
            y_train = features.iloc[train_start:train_end][target_col]
            X_test = features.iloc[train_end:test_end][feature_cols]
            y_test = features.iloc[train_end:test_end][target_col]

            if len(X_test) == 0:
                break

            model = self._create_model(model_type)
            try:
                model.train(X_train, y_train)
                y_pred = model.predict(X_test)
                proba = model.predict_proba(X_test)[:, 1]

                for j, idx in enumerate(X_test.index):
                    if y_test.iloc[j] is not None and not pd.isna(y_test.iloc[j]):
                        results.append({
                            "date": idx,
                            "actual": y_test.iloc[j],
                            "predicted": y_pred[j],
                            "confidence": proba[j],
                            "split": i,
                        })
            except Exception as e:
                logger.warning(f"Split {i} failed: {e}")
                continue

        results_df = pd.DataFrame(results)
        if not results_df.empty:
            from sklearn.metrics import accuracy_score
            acc = accuracy_score(results_df["actual"], results_df["predicted"])
            logger.info(f"Walk-forward accuracy: {acc:.4f} over {len(results_df)} predictions")

        return results_df

    def _create_model(self, model_type: str) -> BaseModel:
        """Factory method for creating models."""
        models: dict[str, type] = {
            "random_forest": RandomForestModel,
            "xgboost": XGBoostModel,
        }

        # Try importing deep learning models (optional dependency - requires PyTorch)
        try:
            from src.models.lstm_model import LSTMModel
            from src.models.transformer_model import TransformerModel
            models["lstm"] = LSTMModel
            models["transformer"] = TransformerModel
        except (ImportError, NameError):
            pass

        if model_type == "ensemble":
            # Create ensemble with all available classical models
            from src.models.ensemble import EnsembleModel
            ensemble = EnsembleModel()
            ensemble.add_model(RandomForestModel())
            ensemble.add_model(XGBoostModel())
            return ensemble

        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
        return models[model_type]()
