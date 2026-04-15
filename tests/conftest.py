"""Shared test fixtures."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.bdate_range(start="2024-01-01", periods=252)
    tickers = ["AAPL", "MSFT"]

    frames = []
    for ticker in tickers:
        base_price = 150 if ticker == "AAPL" else 350
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = base_price * np.cumprod(1 + returns)

        df = pd.DataFrame({
            "open": prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
            "high": prices * (1 + np.abs(np.random.normal(0, 0.015, len(dates)))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.015, len(dates)))),
            "close": prices,
            "volume": np.random.randint(1_000_000, 50_000_000, len(dates)).astype(float),
            "ticker": ticker,
        }, index=dates)
        df.index.name = "date"
        frames.append(df)

    return pd.concat(frames)


@pytest.fixture
def sample_features(sample_ohlcv):
    """Generate sample feature matrix for model testing."""
    df = sample_ohlcv.copy()
    grouped = df.groupby("ticker")

    # Add some basic features
    for window in [5, 10, 20]:
        df[f"sma_{window}"] = grouped["close"].transform(lambda x: x.rolling(window).mean())
        df[f"momentum_{window}"] = grouped["close"].transform(lambda x: x.pct_change(window))

    df["daily_return"] = grouped["close"].transform(lambda x: x.pct_change())
    df["volatility_20"] = grouped["daily_return"].transform(lambda x: x.rolling(20).std())
    df["volume_ratio"] = df["volume"] / grouped["volume"].transform(lambda x: x.rolling(20).mean())

    # RSI
    delta = grouped["close"].transform(lambda x: x.diff())
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta).clip(lower=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Target
    df["target_return_1d"] = grouped["close"].transform(lambda x: x.pct_change().shift(-1))
    df["target_direction"] = (df["target_return_1d"] > 0).astype(int)

    # Add dummy congress and economic features
    df["congress_buy_count"] = 0
    df["congress_sell_count"] = 0
    df["congress_net_signal"] = 0
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month

    return df.dropna(subset=["sma_20", "rsi_14"])
