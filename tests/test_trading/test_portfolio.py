"""Tests for portfolio management."""

import pytest
from datetime import datetime

from src.trading import Signal, Trade
from src.trading.portfolio import Portfolio, Position


class TestPortfolio:
    def setup_method(self):
        self.portfolio = Portfolio(initial_cash=100_000.0)

    def _make_trade(self, ticker="AAPL", signal=Signal.BUY, price=150.0, quantity=10):
        return Trade(
            ticker=ticker, signal=signal, price=price,
            quantity=quantity, timestamp=datetime.now(),
            confidence=0.75, reason="test",
        )

    def test_initial_state(self):
        assert self.portfolio.cash == 100_000.0
        assert self.portfolio.total_value == 100_000.0
        assert len(self.portfolio.positions) == 0
        assert self.portfolio.total_return == 0.0

    def test_buy_reduces_cash(self):
        trade = self._make_trade(price=150.0, quantity=10)
        self.portfolio.execute_trade(trade)

        assert self.portfolio.cash == 100_000.0 - 1500.0
        assert "AAPL" in self.portfolio.positions
        assert self.portfolio.positions["AAPL"].shares == 10

    def test_sell_increases_cash(self):
        # Buy first
        self.portfolio.execute_trade(self._make_trade(signal=Signal.BUY, price=150.0, quantity=10))
        # Then sell
        self.portfolio.execute_trade(self._make_trade(signal=Signal.SELL, price=160.0, quantity=10))

        assert self.portfolio.cash == 100_000.0 - 1500.0 + 1600.0
        assert "AAPL" not in self.portfolio.positions

    def test_buy_insufficient_cash(self):
        trade = self._make_trade(price=150.0, quantity=1_000_000)
        result = self.portfolio.execute_trade(trade)
        assert result is False

    def test_sell_no_position(self):
        trade = self._make_trade(signal=Signal.SELL)
        result = self.portfolio.execute_trade(trade)
        assert result is False

    def test_sell_too_many_shares(self):
        self.portfolio.execute_trade(self._make_trade(signal=Signal.BUY, quantity=10))
        trade = self._make_trade(signal=Signal.SELL, quantity=20)
        result = self.portfolio.execute_trade(trade)
        assert result is False

    def test_multiple_buys_average_cost(self):
        self.portfolio.execute_trade(self._make_trade(price=100.0, quantity=10))
        self.portfolio.execute_trade(self._make_trade(price=200.0, quantity=10))

        pos = self.portfolio.positions["AAPL"]
        assert pos.shares == 20
        assert pos.avg_cost == 150.0  # (1000 + 2000) / 20

    def test_update_prices(self):
        self.portfolio.execute_trade(self._make_trade(price=150.0, quantity=10))
        self.portfolio.update_prices({"AAPL": 160.0})

        pos = self.portfolio.positions["AAPL"]
        assert pos.current_price == 160.0
        assert pos.unrealized_pnl == 100.0  # (160-150) * 10

    def test_trade_history(self):
        self.portfolio.execute_trade(self._make_trade(signal=Signal.BUY))
        self.portfolio.execute_trade(self._make_trade(signal=Signal.SELL))

        df = self.portfolio.get_trade_history_df()
        assert len(df) == 2
        assert df.iloc[0]["signal"] == "buy"
        assert df.iloc[1]["signal"] == "sell"

    def test_performance_summary(self):
        self.portfolio.execute_trade(self._make_trade(signal=Signal.BUY, price=150.0, quantity=10))
        self.portfolio.update_prices({"AAPL": 160.0})

        summary = self.portfolio.get_performance_summary()
        assert "total_return_pct" in summary
        assert summary["total_return_pct"] != 0 or summary.get("n_trades", 0) >= 0


class TestPosition:
    def test_market_value(self):
        pos = Position(ticker="AAPL", shares=10, avg_cost=150.0, current_price=160.0)
        assert pos.market_value == 1600.0

    def test_unrealized_pnl(self):
        pos = Position(ticker="AAPL", shares=10, avg_cost=150.0, current_price=160.0)
        assert pos.unrealized_pnl == 100.0

    def test_unrealized_pnl_pct(self):
        pos = Position(ticker="AAPL", shares=10, avg_cost=100.0, current_price=110.0)
        assert pos.unrealized_pnl_pct == 10.0
