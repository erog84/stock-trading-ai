"""Model API routes."""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel as PydanticModel
from typing import Optional
from pathlib import Path

from src.data.aggregator import DataAggregator
from src.models.model_trainer import ModelTrainer

router = APIRouter()

_aggregator = DataAggregator()
_trainer = ModelTrainer()

# Track training status
_training_status = {"status": "idle", "metrics": None, "error": None}


class TrainRequest(PydanticModel):
    tickers: list[str]
    start_date: str
    end_date: Optional[str] = None
    model_type: str = "random_forest"


def _train_model_task(request: TrainRequest):
    """Background task for model training."""
    global _training_status
    _training_status = {"status": "training", "metrics": None, "error": None}

    try:
        # Build features
        features = _aggregator.fetch_and_build_features(
            request.tickers, request.start_date, request.end_date,
        )
        _aggregator.save_features(features)

        # Train model
        model, metrics = _trainer.train_model(
            features, model_type=request.model_type,
        )
        _training_status = {"status": "complete", "metrics": metrics, "error": None}

    except Exception as e:
        _training_status = {"status": "error", "metrics": None, "error": str(e)}


@router.post("/train")
def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
    """Start model training (runs in background)."""
    if _training_status["status"] == "training":
        raise HTTPException(status_code=409, detail="Training already in progress")

    background_tasks.add_task(_train_model_task, request)
    return {"message": "Training started", "model_type": request.model_type}


@router.get("/status")
def get_training_status():
    """Get current training status."""
    return _training_status


@router.get("/metrics")
def get_model_metrics():
    """Get latest model metrics."""
    if _training_status["metrics"] is None:
        raise HTTPException(status_code=404, detail="No model trained yet")
    return _training_status["metrics"]


@router.get("/feature-importance")
def get_feature_importance():
    """Get feature importance from latest model."""
    model_path = Path("data/models/random_forest_latest.joblib")
    if not model_path.exists():
        model_path = Path("data/models/xgboost_latest.joblib")
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="No model found")

    from src.models.random_forest import RandomForestModel
    model = RandomForestModel()
    model.load(str(model_path))
    importance = model.get_feature_importance()

    # Return top 20
    return dict(list(importance.items())[:20])
