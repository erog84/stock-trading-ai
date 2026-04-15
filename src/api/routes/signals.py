"""Trading signals API routes."""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from pathlib import Path

from src.data.yahoo_finance import YahooFinanceFetcher
from src.data.aggregator import DataAggregator
from src.models.random_forest import RandomForestModel
from src.trading.signals import SignalGenerator

router = APIRouter()


@router.get("/latest")
def get_latest_signals(
    tickers: str = Query(..., description="Comma-separated ticker symbols"),
    confidence: float = Query(0.6, description="Minimum confidence threshold"),
):
    """Generate latest trading signals for given tickers."""
    ticker_list = [t.strip().upper() for t in tickers.split(",")]

    # Load model
    model_path = Path("data/models/random_forest_latest.joblib")
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="No trained model. Train a model first via POST /api/models/train")

    model = RandomForestModel()
    model.load(str(model_path))

    # Get latest features
    aggregator = DataAggregator()
    yahoo = YahooFinanceFetcher()

    try:
        features = aggregator.fetch_and_build_features(
            ticker_list,
            start_date="2024-01-01",  # Need history for indicators
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error building features: {e}")

    if features.empty:
        return []

    # Get latest data point per ticker
    latest = features.groupby("ticker").tail(1)

    # Get current prices
    current_prices = {
        row["ticker"]: row["close"]
        for _, row in latest.iterrows()
    }

    # Generate signals
    generator = SignalGenerator(model, confidence_threshold=confidence)
    signals = generator.generate_signals(
        features=latest,
        current_prices=current_prices,
        portfolio_value=100_000,
        current_positions=set(),
    )

    return [
        {
            "ticker": s.ticker,
            "signal": s.signal.value,
            "price": s.price,
            "quantity": s.quantity,
            "confidence": s.confidence,
            "reason": s.reason,
        }
        for s in signals
    ]
