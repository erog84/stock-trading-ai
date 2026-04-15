"""Tests for Random Forest model."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from src.models.random_forest import RandomForestModel


class TestRandomForestModel:
    def test_train_returns_metrics(self, sample_features):
        model = RandomForestModel(n_estimators=10, random_state=42)
        X = sample_features.drop(columns=["target_return_1d", "target_direction", "ticker", "open", "high", "low", "close", "volume"])
        y = sample_features["target_direction"]

        metrics = model.train(X, y)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "f1" in metrics
        assert metrics["accuracy"] > 0
        assert metrics["n_samples"] > 0

    def test_predict_returns_array(self, sample_features):
        model = RandomForestModel(n_estimators=10, random_state=42)
        X = sample_features.drop(columns=["target_return_1d", "target_direction", "ticker", "open", "high", "low", "close", "volume"])
        y = sample_features["target_direction"]

        model.train(X, y)
        predictions = model.predict(X.head(10))

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)

    def test_predict_proba(self, sample_features):
        model = RandomForestModel(n_estimators=10, random_state=42)
        X = sample_features.drop(columns=["target_return_1d", "target_direction", "ticker", "open", "high", "low", "close", "volume"])
        y = sample_features["target_direction"]

        model.train(X, y)
        proba = model.predict_proba(X.head(5))

        assert proba.shape == (5, 2)
        assert all(0 <= p <= 1 for row in proba for p in row)

    def test_feature_importance(self, sample_features):
        model = RandomForestModel(n_estimators=10, random_state=42)
        X = sample_features.drop(columns=["target_return_1d", "target_direction", "ticker", "open", "high", "low", "close", "volume"])
        y = sample_features["target_direction"]

        model.train(X, y)
        importance = model.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) > 0
        # Should be sorted by importance
        values = list(importance.values())
        assert values == sorted(values, reverse=True)

    def test_save_and_load(self, sample_features):
        model = RandomForestModel(n_estimators=10, random_state=42)
        X = sample_features.drop(columns=["target_return_1d", "target_direction", "ticker", "open", "high", "low", "close", "volume"])
        y = sample_features["target_direction"]

        model.train(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "model.joblib")
            model.save(path)

            model2 = RandomForestModel()
            model2.load(path)

            # Predictions should be identical
            pred1 = model.predict(X.head(5))
            pred2 = model2.predict(X.head(5))
            np.testing.assert_array_equal(pred1, pred2)

    def test_insufficient_data(self):
        model = RandomForestModel()
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = pd.Series([0, 1, 0])

        with pytest.raises(ValueError, match="Not enough training data"):
            model.train(X, y)
