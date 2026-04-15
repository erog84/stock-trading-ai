"""Historical backtesting engine."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np

from src.models import BaseModel
from src.models.model_trainer import ModelTrainer, get_feature_columns
from src.trading.portfolio import Portfolio
from src.trading.signals import SignalGenerator
from src.trading import Signal, Trade
from src.utils.logger import logger


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    initial_cash: float = 100_000.0
    confidence_threshold: float = 0.6
    position_size_pct: float = 0.05
    max_positions: int = 20
    commission_per_trade: float = 0.0
    slippage_pct: float = 0.001


class Backtester:
    """Run historical backtests on trading strategies."""

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        features: pd.DataFrame,
        model: BaseModel,
        train_window: int = 252,
    ) -> dict:
        """Run a full backtest with walk-forward training.

        Args:
            features: Full feature matrix with targets
            model: Model to use for predictions
            train_window: Number of days to train on before trading

        Returns:
            Dictionary with backtest results
        """
        portfolio = Portfolio(initial_cash=self.config.initial_cash)
        feature_cols = get_feature_columns(features)
        trades_log = []

        # Get unique dates
        dates = features.index.unique().sort_values()

        # Adjust train_window if we don't have enough unique trading days
        if len(dates) < train_window + 21:
            train_window = max(50, len(dates) - 21)
            logger.warning(f"Not enough trading days for full train window. Reduced to {train_window} days.")

        if len(dates) < train_window + 5:
            raise ValueError(f"Not enough trading days for backtest: {len(dates)} days, need at least {train_window + 5}")

        logger.info(f"Running backtest from {dates[train_window]} to {dates[-1]} ({len(dates)} trading days, train_window={train_window})")

        # Walk through each trading day after initial training period
        for i in range(train_window, len(dates)):
            current_date = dates[i]
            train_data = features.loc[dates[:i]]
            today_data = features.loc[[current_date]]

            # Retrain model periodically (every 21 trading days)
            if (i - train_window) % 21 == 0:
                try:
                    X_train = train_data[feature_cols].iloc[-train_window:]
                    y_train = train_data["target_direction"].iloc[-train_window:]
                    model.train(X_train, y_train)
                except Exception as e:
                    logger.debug(f"Retrain failed at {current_date}: {e}")
                    continue

            # Get predictions for today
            try:
                proba = model.predict_proba(today_data[feature_cols])
            except Exception:
                continue

            # Process each ticker for today
            for j, (idx, row) in enumerate(today_data.iterrows()):
                ticker = row.get("ticker", "UNKNOWN")
                price = row["close"]
                confidence = proba[j][1]

                # Apply slippage
                buy_price = price * (1 + self.config.slippage_pct)
                sell_price = price * (1 - self.config.slippage_pct)

                # Generate signals
                if confidence >= self.config.confidence_threshold:
                    if ticker not in portfolio.positions and len(portfolio.positions) < self.config.max_positions:
                        # BUY
                        position_value = portfolio.total_value * self.config.position_size_pct
                        quantity = max(1, int(position_value / buy_price))

                        trade = Trade(
                            ticker=ticker, signal=Signal.BUY, price=buy_price,
                            quantity=quantity, timestamp=current_date,
                            confidence=confidence, reason=f"Model: {confidence:.1%} up",
                        )
                        if portfolio.execute_trade(trade):
                            portfolio.cash -= self.config.commission_per_trade
                            trades_log.append(trade)

                elif (1 - confidence) >= self.config.confidence_threshold:
                    if ticker in portfolio.positions:
                        # SELL all shares
                        shares = portfolio.positions[ticker].shares
                        trade = Trade(
                            ticker=ticker, signal=Signal.SELL, price=sell_price,
                            quantity=shares, timestamp=current_date,
                            confidence=1 - confidence, reason=f"Model: {1-confidence:.1%} down",
                        )
                        if portfolio.execute_trade(trade):
                            portfolio.cash -= self.config.commission_per_trade
                            trades_log.append(trade)

            # Update prices and record daily value
            prices = {row["ticker"]: row["close"] for _, row in today_data.iterrows()}
            portfolio.update_prices(prices)
            portfolio.record_daily_value(current_date)

        # Calculate results
        results = self._compile_results(portfolio, trades_log, dates[train_window], dates[-1])
        return results

    def _compile_results(
        self,
        portfolio: Portfolio,
        trades: list[Trade],
        start_date,
        end_date,
    ) -> dict:
        """Compile backtest results into a summary."""
        perf = portfolio.get_performance_summary()
        daily_df = pd.DataFrame(portfolio.daily_values)

        results = {
            "period": f"{start_date} to {end_date}",
            "initial_cash": portfolio.initial_cash,
            "final_value": portfolio.total_value,
            **perf,
            "total_trades": len(trades),
            "buy_trades": sum(1 for t in trades if t.signal == Signal.BUY),
            "sell_trades": sum(1 for t in trades if t.signal == Signal.SELL),
            "daily_values": daily_df,
            "trades": pd.DataFrame([{
                "date": t.timestamp,
                "ticker": t.ticker,
                "signal": t.signal.value,
                "price": t.price,
                "quantity": t.quantity,
                "confidence": t.confidence,
                "reason": t.reason,
            } for t in trades]) if trades else pd.DataFrame(),
        }

        logger.info(
            f"Backtest complete: Return={perf.get('total_return_pct', 0):.2f}%, "
            f"Sharpe={perf.get('sharpe_ratio', 0):.2f}, "
            f"MaxDD={perf.get('max_drawdown_pct', 0):.2f}%, "
            f"Trades={len(trades)}"
        )

        return results
