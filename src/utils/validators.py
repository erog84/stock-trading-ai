"""Data and signal validation utilities."""

import pandas as pd
import numpy as np
from typing import Optional

from src.trading import Trade, Signal
from src.utils.logger import logger


def validate_ohlcv(df: pd.DataFrame) -> list[str]:
    """Validate OHLCV DataFrame. Returns list of issues found."""
    issues = []

    required_cols = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")
        return issues  # Can't validate further

    # Check for NaN close prices
    nan_close = df["close"].isna().sum()
    if nan_close > 0:
        issues.append(f"{nan_close} NaN close prices")

    # Check for negative prices
    for col in ["open", "high", "low", "close"]:
        neg = (df[col] < 0).sum()
        if neg > 0:
            issues.append(f"{neg} negative {col} prices")

    # Check for negative volume
    neg_vol = (df["volume"] < 0).sum()
    if neg_vol > 0:
        issues.append(f"{neg_vol} negative volumes")

    # Check high >= low
    invalid_hl = (df["high"] < df["low"]).sum()
    if invalid_hl > 0:
        issues.append(f"{invalid_hl} rows where high < low")

    # Check for zero-volume days (suspicious but not always invalid)
    zero_vol = (df["volume"] == 0).sum()
    if zero_vol > len(df) * 0.1:
        issues.append(f"{zero_vol} zero-volume days ({zero_vol/len(df)*100:.0f}%)")

    return issues


def validate_features(df: pd.DataFrame, max_nan_pct: float = 0.3) -> list[str]:
    """Validate feature DataFrame. Returns list of issues."""
    issues = []

    if df.empty:
        issues.append("Empty DataFrame")
        return issues

    # NaN percentage per column
    nan_pcts = df.isna().mean()
    high_nan_cols = nan_pcts[nan_pcts > max_nan_pct].index.tolist()
    if high_nan_cols:
        issues.append(f"Columns with >{max_nan_pct*100:.0f}% NaN: {high_nan_cols}")

    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number])
    inf_cols = numeric_cols.columns[numeric_cols.isin([np.inf, -np.inf]).any()].tolist()
    if inf_cols:
        issues.append(f"Columns with infinite values: {inf_cols}")

    # Check date index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" not in df.columns:
            issues.append("No DatetimeIndex or date column")

    return issues


def validate_signal(trade: Trade) -> list[str]:
    """Validate a trading signal. Returns list of issues."""
    issues = []

    if not trade.ticker or not trade.ticker.isalpha():
        issues.append(f"Invalid ticker: {trade.ticker}")

    if trade.quantity <= 0:
        issues.append(f"Invalid quantity: {trade.quantity}")

    if trade.price <= 0:
        issues.append(f"Invalid price: {trade.price}")

    if not 0 <= trade.confidence <= 1:
        issues.append(f"Confidence out of range [0,1]: {trade.confidence}")

    if trade.signal not in (Signal.BUY, Signal.SELL, Signal.HOLD):
        issues.append(f"Invalid signal: {trade.signal}")

    return issues


def validate_and_log(df: pd.DataFrame, source: str, validator: str = "ohlcv") -> bool:
    """Validate data and log any issues. Returns True if data is clean."""
    if validator == "ohlcv":
        issues = validate_ohlcv(df)
    else:
        issues = validate_features(df)

    if issues:
        for issue in issues:
            logger.warning(f"Data validation [{source}]: {issue}")
        return False

    logger.debug(f"Data validation [{source}]: OK ({len(df)} rows)")
    return True
