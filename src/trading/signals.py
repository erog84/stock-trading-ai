"""Buy/sell signal generation from model predictions."""

from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np

from src.models import BaseModel
from src.trading import Signal, Trade
from src.utils.logger import logger


class SignalGenerator:
    """Generates trading signals from model predictions and rules."""

    def __init__(
        self,
        model: BaseModel,
        confidence_threshold: float = 0.6,
        position_size_pct: float = 0.05,  # Max 5% of portfolio per position
        max_positions: int = 20,
    ):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions

    def generate_signals(
        self,
        features: pd.DataFrame,
        current_prices: dict[str, float],
        portfolio_value: float,
        current_positions: set[str],
    ) -> list[Trade]:
        """Generate buy/sell signals for the latest data point.

        Args:
            features: Feature matrix with latest data
            current_prices: Current price for each ticker
            portfolio_value: Total portfolio value for position sizing
            current_positions: Set of tickers currently held

        Returns:
            List of Trade objects (signals)
        """
        trades = []

        # Get predictions and probabilities
        try:
            predictions = self.model.predict(features)
            probabilities = self.model.predict_proba(features)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return trades

        # Process each ticker
        for i, (idx, row) in enumerate(features.iterrows()):
            ticker = row.get("ticker", "UNKNOWN")
            if ticker not in current_prices:
                continue

            pred = predictions[i]
            confidence = probabilities[i][1]  # Probability of class 1 (up)
            price = current_prices[ticker]

            # Generate signal based on prediction and confidence
            signal = self._evaluate_signal(
                ticker=ticker,
                prediction=pred,
                confidence=confidence,
                price=price,
                portfolio_value=portfolio_value,
                is_held=ticker in current_positions,
                features=row,
            )

            if signal is not None:
                trades.append(signal)

        # Sort by confidence (highest first)
        trades.sort(key=lambda t: t.confidence, reverse=True)

        # Limit number of new buy signals
        buy_count = sum(1 for t in trades if t.signal == Signal.BUY)
        available_slots = self.max_positions - len(current_positions)
        if buy_count > available_slots:
            buys = [t for t in trades if t.signal == Signal.BUY][:available_slots]
            sells = [t for t in trades if t.signal == Signal.SELL]
            trades = sells + buys

        return trades

    def _evaluate_signal(
        self,
        ticker: str,
        prediction: int,
        confidence: float,
        price: float,
        portfolio_value: float,
        is_held: bool,
        features: pd.Series,
    ) -> Optional[Trade]:
        """Evaluate whether to generate a trade signal."""
        now = datetime.now()

        # BUY signal: model predicts up with high confidence
        if prediction == 1 and confidence >= self.confidence_threshold and not is_held:
            # Position sizing
            position_value = portfolio_value * self.position_size_pct
            quantity = max(1, int(position_value / price))

            reason = self._build_reason(features, "BUY", confidence)

            return Trade(
                ticker=ticker,
                signal=Signal.BUY,
                price=price,
                quantity=quantity,
                timestamp=now,
                confidence=confidence,
                reason=reason,
            )

        # SELL signal: model predicts down with high confidence on held position
        elif prediction == 0 and (1 - confidence) >= self.confidence_threshold and is_held:
            reason = self._build_reason(features, "SELL", 1 - confidence)

            return Trade(
                ticker=ticker,
                signal=Signal.SELL,
                price=price,
                quantity=0,  # Will be set to full position by executor
                timestamp=now,
                confidence=1 - confidence,
                reason=reason,
            )

        return None

    def _build_reason(self, features: pd.Series, action: str, confidence: float) -> str:
        """Build a human-readable reason for the signal."""
        reasons = [f"Model confidence: {confidence:.1%}"]

        # Add technical indicator context
        if "rsi_14" in features and not pd.isna(features["rsi_14"]):
            rsi = features["rsi_14"]
            if rsi < 30:
                reasons.append(f"RSI oversold ({rsi:.0f})")
            elif rsi > 70:
                reasons.append(f"RSI overbought ({rsi:.0f})")

        if "macd_hist" in features and not pd.isna(features["macd_hist"]):
            if features["macd_hist"] > 0:
                reasons.append("MACD bullish")
            else:
                reasons.append("MACD bearish")

        if "congress_activity_30d" in features and features.get("congress_activity_30d", 0) > 0:
            reasons.append(f"Congress buying activity ({int(features['congress_activity_30d'])} net)")

        return "; ".join(reasons)
