"""Trade execution engine with risk management.

Routes signals through the configured broker with safety checks.
"""

from datetime import datetime, time
from typing import Optional
import pytz

from src.trading import Signal, Trade
from src.trading.broker_api import BrokerAPI, PaperBroker, OrderStatus
from src.trading.alpaca_broker import AlpacaBroker
from src.trading.portfolio import Portfolio
from src.utils.config import config
from src.utils.logger import logger


class RiskLimits:
    """Risk management parameters."""

    def __init__(
        self,
        max_position_pct: float = 0.10,       # Max 10% of portfolio in one position
        max_daily_loss_pct: float = 0.03,      # Stop trading if down 3% today
        max_open_positions: int = 20,
        min_cash_reserve_pct: float = 0.10,    # Keep 10% cash minimum
        require_market_hours: bool = True,
    ):
        self.max_position_pct = max_position_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_open_positions = max_open_positions
        self.min_cash_reserve_pct = min_cash_reserve_pct
        self.require_market_hours = require_market_hours


class TradeExecutor:
    """Executes trading signals through a broker with risk checks."""

    def __init__(
        self,
        broker: Optional[BrokerAPI] = None,
        risk_limits: Optional[RiskLimits] = None,
    ):
        self.broker = broker or self._create_broker()
        self.risk_limits = risk_limits or RiskLimits()
        self.daily_start_value: Optional[float] = None
        self.executed_today: list[dict] = []

    def _create_broker(self) -> BrokerAPI:
        """Create broker based on configuration."""
        broker_type = config.broker_type
        if broker_type in ("alpaca_paper", "alpaca_live"):
            paper = broker_type == "alpaca_paper"
            return AlpacaBroker(paper=paper)
        return PaperBroker()

    def execute_signals(self, signals: list[Trade]) -> list[dict]:
        """Execute a list of trading signals with risk checks.

        Returns list of execution results.
        """
        results = []

        # Get account info for risk checks
        try:
            account = self.broker.get_account_info()
        except Exception as e:
            logger.error(f"Cannot get account info: {e}")
            return [{"error": str(e), "signals_skipped": len(signals)}]

        total_value = account["total_value"]
        cash = account["cash"]

        # Set daily start value on first call of the day
        if self.daily_start_value is None:
            self.daily_start_value = total_value

        # Check daily loss limit
        daily_pnl_pct = (total_value - self.daily_start_value) / self.daily_start_value
        if daily_pnl_pct < -self.risk_limits.max_daily_loss_pct:
            msg = f"Daily loss limit hit ({daily_pnl_pct:.2%}). Halting trading."
            logger.warning(msg)
            return [{"error": msg, "signals_skipped": len(signals)}]

        # Check market hours
        if self.risk_limits.require_market_hours and not self._is_market_open():
            msg = "Market is closed. Skipping signal execution."
            logger.info(msg)
            return [{"error": msg, "signals_skipped": len(signals)}]

        # Get current positions
        try:
            positions = self.broker.get_positions()
            held_tickers = {p["ticker"] for p in positions}
            n_positions = len(positions)
        except Exception as e:
            logger.error(f"Cannot get positions: {e}")
            return [{"error": str(e)}]

        # Process sells first (free up capital), then buys
        sells = [s for s in signals if s.signal == Signal.SELL]
        buys = [s for s in signals if s.signal == Signal.BUY]

        for trade in sells + buys:
            result = self._execute_single(trade, total_value, cash, held_tickers, n_positions)
            results.append(result)

            # Update tracking
            if result.get("status") == "filled":
                if trade.signal == Signal.BUY:
                    cash -= trade.price * trade.quantity
                    held_tickers.add(trade.ticker)
                    n_positions += 1
                elif trade.signal == Signal.SELL:
                    cash += trade.price * trade.quantity
                    held_tickers.discard(trade.ticker)
                    n_positions -= 1

        self.executed_today.extend(results)
        return results

    def _execute_single(
        self,
        trade: Trade,
        total_value: float,
        cash: float,
        held_tickers: set,
        n_positions: int,
    ) -> dict:
        """Execute a single trade with risk checks."""
        result = {
            "ticker": trade.ticker,
            "signal": trade.signal.value,
            "quantity": trade.quantity,
            "price": trade.price,
            "confidence": trade.confidence,
        }

        # Risk checks for buys
        if trade.signal == Signal.BUY:
            # Max positions check
            if n_positions >= self.risk_limits.max_open_positions:
                result["status"] = "rejected"
                result["reason"] = f"Max positions ({self.risk_limits.max_open_positions}) reached"
                logger.warning(f"Rejected {trade.ticker}: {result['reason']}")
                return result

            # Position size check
            position_value = trade.price * trade.quantity
            if position_value / total_value > self.risk_limits.max_position_pct:
                # Reduce quantity to fit within limits
                max_value = total_value * self.risk_limits.max_position_pct
                trade.quantity = max(1, int(max_value / trade.price))
                result["quantity"] = trade.quantity
                logger.info(f"Reduced {trade.ticker} quantity to {trade.quantity} for position size limit")

            # Cash reserve check
            position_value = trade.price * trade.quantity
            min_cash = total_value * self.risk_limits.min_cash_reserve_pct
            if cash - position_value < min_cash:
                result["status"] = "rejected"
                result["reason"] = "Would breach cash reserve minimum"
                logger.warning(f"Rejected {trade.ticker}: {result['reason']}")
                return result

            # Already holding check
            if trade.ticker in held_tickers:
                result["status"] = "rejected"
                result["reason"] = "Already holding position"
                return result

        # Risk checks for sells
        if trade.signal == Signal.SELL and trade.ticker not in held_tickers:
            result["status"] = "rejected"
            result["reason"] = "No position to sell"
            return result

        # Submit order
        try:
            side = "buy" if trade.signal == Signal.BUY else "sell"
            order = self.broker.submit_order(
                ticker=trade.ticker,
                side=side,
                quantity=trade.quantity,
            )
            result["order_id"] = order.order_id
            result["status"] = order.status.value
            result["filled_price"] = order.filled_price
            logger.info(f"Order {order.order_id}: {side.upper()} {trade.quantity} {trade.ticker} -> {order.status.value}")

        except Exception as e:
            result["status"] = "error"
            result["reason"] = str(e)
            logger.error(f"Execution error for {trade.ticker}: {e}")

        return result

    def _is_market_open(self) -> bool:
        """Check if US stock market is currently open."""
        et = pytz.timezone("US/Eastern")
        now = datetime.now(et)

        # Weekend check
        if now.weekday() >= 5:
            return False

        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = time(9, 30)
        market_close = time(16, 0)
        return market_open <= now.time() <= market_close

    def reset_daily_tracking(self) -> None:
        """Reset daily tracking (call at start of each trading day)."""
        self.daily_start_value = None
        self.executed_today = []
