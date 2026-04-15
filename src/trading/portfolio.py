"""Portfolio tracking and management."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import pandas as pd

from src.trading import Signal, Trade
from src.utils.logger import logger


@dataclass
class Position:
    """Represents a current holding."""
    ticker: str
    shares: int
    avg_cost: float
    current_price: float = 0.0
    opened_at: datetime = field(default_factory=datetime.now)

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.shares * self.avg_cost

    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_basis

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl / self.cost_basis) * 100


class Portfolio:
    """Tracks positions, trades, and performance."""

    def __init__(self, initial_cash: float = 100_000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: dict[str, Position] = {}
        self.trade_history: list[Trade] = []
        self.daily_values: list[dict] = []

    @property
    def total_value(self) -> float:
        positions_value = sum(p.market_value for p in self.positions.values())
        return self.cash + positions_value

    @property
    def total_return(self) -> float:
        return (self.total_value - self.initial_cash) / self.initial_cash * 100

    def execute_trade(self, trade: Trade) -> bool:
        """Execute a trade and update portfolio state."""
        if trade.signal == Signal.BUY:
            return self._buy(trade)
        elif trade.signal == Signal.SELL:
            return self._sell(trade)
        return False

    def _buy(self, trade: Trade) -> bool:
        """Execute a buy order."""
        cost = trade.price * trade.quantity
        if cost > self.cash:
            logger.warning(f"Insufficient cash for {trade.ticker}: need ${cost:.2f}, have ${self.cash:.2f}")
            return False

        self.cash -= cost

        if trade.ticker in self.positions:
            pos = self.positions[trade.ticker]
            total_shares = pos.shares + trade.quantity
            pos.avg_cost = (pos.cost_basis + cost) / total_shares
            pos.shares = total_shares
        else:
            self.positions[trade.ticker] = Position(
                ticker=trade.ticker,
                shares=trade.quantity,
                avg_cost=trade.price,
                current_price=trade.price,
                opened_at=trade.timestamp,
            )

        self.trade_history.append(trade)
        logger.info(f"BUY {trade.quantity} {trade.ticker} @ ${trade.price:.2f} (confidence: {trade.confidence:.2f})")
        return True

    def _sell(self, trade: Trade) -> bool:
        """Execute a sell order."""
        if trade.ticker not in self.positions:
            logger.warning(f"No position in {trade.ticker} to sell")
            return False

        pos = self.positions[trade.ticker]
        if trade.quantity > pos.shares:
            logger.warning(f"Cannot sell {trade.quantity} shares of {trade.ticker}, only have {pos.shares}")
            return False

        proceeds = trade.price * trade.quantity
        self.cash += proceeds

        pos.shares -= trade.quantity
        if pos.shares == 0:
            del self.positions[trade.ticker]

        self.trade_history.append(trade)
        logger.info(f"SELL {trade.quantity} {trade.ticker} @ ${trade.price:.2f} (confidence: {trade.confidence:.2f})")
        return True

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update current prices for all positions."""
        for ticker, price in prices.items():
            if ticker in self.positions:
                self.positions[ticker].current_price = price

    def record_daily_value(self, date: datetime) -> None:
        """Record portfolio value for performance tracking."""
        self.daily_values.append({
            "date": date,
            "total_value": self.total_value,
            "cash": self.cash,
            "positions_value": self.total_value - self.cash,
            "n_positions": len(self.positions),
        })

    def get_performance_summary(self) -> dict:
        """Calculate portfolio performance metrics."""
        if not self.daily_values:
            return {
                "total_return_pct": self.total_return,
                "total_value": self.total_value,
                "cash": self.cash,
                "n_positions": len(self.positions),
                "n_trades": len(self.trade_history),
            }

        df = pd.DataFrame(self.daily_values).set_index("date")
        returns = df["total_value"].pct_change().dropna()

        # Metrics
        total_return = self.total_return
        n_trades = len(self.trade_history)
        winning_trades = sum(
            1 for t in self.trade_history
            if t.signal == Signal.SELL and t.price > self._get_avg_cost_at_trade(t)
        )
        losing_trades = sum(
            1 for t in self.trade_history
            if t.signal == Signal.SELL and t.price <= self._get_avg_cost_at_trade(t)
        )

        summary = {
            "total_return_pct": total_return,
            "total_value": self.total_value,
            "cash": self.cash,
            "n_positions": len(self.positions),
            "n_trades": n_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": winning_trades / max(winning_trades + losing_trades, 1) * 100,
        }

        if len(returns) > 1:
            import numpy as np
            summary["sharpe_ratio"] = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
            summary["max_drawdown_pct"] = self._calculate_max_drawdown(df["total_value"])
            summary["volatility"] = returns.std() * np.sqrt(252) * 100

        return summary

    def _get_avg_cost_at_trade(self, trade: Trade) -> float:
        """Get average cost basis at time of trade (approximation)."""
        # Look through trade history for buys of this ticker before this sell
        buys = [
            t for t in self.trade_history
            if t.ticker == trade.ticker and t.signal == Signal.BUY and t.timestamp <= trade.timestamp
        ]
        if not buys:
            return trade.price
        total_cost = sum(t.price * t.quantity for t in buys)
        total_shares = sum(t.quantity for t in buys)
        return total_cost / total_shares if total_shares > 0 else trade.price

    def _calculate_max_drawdown(self, values: pd.Series) -> float:
        """Calculate maximum drawdown percentage."""
        peak = values.expanding().max()
        drawdown = (values - peak) / peak * 100
        return drawdown.min()

    def get_trade_history_df(self) -> pd.DataFrame:
        """Return trade history as a DataFrame."""
        if not self.trade_history:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "timestamp": t.timestamp,
                "ticker": t.ticker,
                "signal": t.signal.value,
                "price": t.price,
                "quantity": t.quantity,
                "confidence": t.confidence,
                "reason": t.reason,
            }
            for t in self.trade_history
        ])
