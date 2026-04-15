"""Tests for trade executor with risk management."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from src.trading import Signal, Trade
from src.trading.executor import TradeExecutor, RiskLimits
from src.trading.broker_api import PaperBroker, OrderStatus


class TestTradeExecutor:
    def setup_method(self):
        self.broker = PaperBroker(initial_cash=100_000)
        self.broker.set_prices({"AAPL": 150.0, "MSFT": 350.0, "GOOGL": 140.0})
        self.executor = TradeExecutor(
            broker=self.broker,
            risk_limits=RiskLimits(require_market_hours=False),
        )

    def _make_trade(self, ticker="AAPL", signal=Signal.BUY, price=150.0, quantity=10, confidence=0.75):
        return Trade(
            ticker=ticker, signal=signal, price=price,
            quantity=quantity, timestamp=datetime.now(),
            confidence=confidence, reason="test",
        )

    def test_execute_buy_signal(self):
        signals = [self._make_trade()]
        results = self.executor.execute_signals(signals)
        assert len(results) == 1
        assert results[0]["status"] == "filled"

    def test_execute_sell_requires_position(self):
        signals = [self._make_trade(signal=Signal.SELL)]
        results = self.executor.execute_signals(signals)
        assert results[0]["status"] == "rejected"
        assert "No position" in results[0]["reason"]

    def test_max_positions_limit(self):
        self.executor.risk_limits.max_open_positions = 1
        signals = [
            self._make_trade(ticker="AAPL"),
            self._make_trade(ticker="MSFT", price=350.0),
        ]
        results = self.executor.execute_signals(signals)
        filled = [r for r in results if r["status"] == "filled"]
        rejected = [r for r in results if r["status"] == "rejected"]
        assert len(filled) == 1
        assert len(rejected) == 1

    def test_daily_loss_limit(self):
        self.executor.daily_start_value = 100_000
        # Simulate loss by reducing cash
        self.broker.cash = 96_000  # 4% loss
        self.executor.risk_limits.max_daily_loss_pct = 0.03

        signals = [self._make_trade()]
        results = self.executor.execute_signals(signals)
        assert "Daily loss limit" in str(results)

    def test_cash_reserve_check(self):
        self.executor.risk_limits.min_cash_reserve_pct = 0.95  # Need 95% cash reserve
        signals = [self._make_trade(price=150.0, quantity=100)]
        results = self.executor.execute_signals(signals)
        assert results[0]["status"] == "rejected"

    def test_sells_execute_before_buys(self):
        # Buy first
        self.broker.submit_order("AAPL", "buy", 10)
        self.broker.set_prices({"AAPL": 160.0, "MSFT": 350.0})

        signals = [
            self._make_trade(ticker="MSFT", signal=Signal.BUY, price=350.0, quantity=5),
            self._make_trade(ticker="AAPL", signal=Signal.SELL, price=160.0, quantity=10),
        ]
        results = self.executor.execute_signals(signals)
        # Both should succeed (sell frees cash for buy)
        assert all(r["status"] == "filled" for r in results)
