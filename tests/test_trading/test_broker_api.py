"""Tests for paper broker API."""

import pytest

from src.trading.broker_api import PaperBroker, OrderStatus


class TestPaperBroker:
    def setup_method(self):
        self.broker = PaperBroker(initial_cash=100_000.0)
        self.broker.set_prices({"AAPL": 150.0, "MSFT": 350.0})

    def test_buy_order(self):
        order = self.broker.submit_order("AAPL", "buy", 10)
        assert order.status == OrderStatus.FILLED
        assert order.filled_price == 150.0
        assert self.broker.cash == 100_000.0 - 1500.0

    def test_sell_order(self):
        self.broker.submit_order("AAPL", "buy", 10)
        order = self.broker.submit_order("AAPL", "sell", 5)
        assert order.status == OrderStatus.FILLED
        assert self.broker.positions["AAPL"]["shares"] == 5

    def test_sell_all_removes_position(self):
        self.broker.submit_order("AAPL", "buy", 10)
        self.broker.submit_order("AAPL", "sell", 10)
        assert "AAPL" not in self.broker.positions

    def test_insufficient_funds(self):
        order = self.broker.submit_order("AAPL", "buy", 1_000_000)
        assert order.status == OrderStatus.REJECTED

    def test_insufficient_shares(self):
        order = self.broker.submit_order("AAPL", "sell", 10)
        assert order.status == OrderStatus.REJECTED

    def test_no_price_available(self):
        order = self.broker.submit_order("UNKNOWN", "buy", 10)
        assert order.status == OrderStatus.REJECTED

    def test_get_account_info(self):
        self.broker.submit_order("AAPL", "buy", 10)
        info = self.broker.get_account_info()
        assert info["cash"] == 100_000.0 - 1500.0
        assert info["positions_value"] == 1500.0
        assert info["total_value"] == 100_000.0
        assert info["n_positions"] == 1

    def test_get_positions(self):
        self.broker.submit_order("AAPL", "buy", 10)
        positions = self.broker.get_positions()
        assert len(positions) == 1
        assert positions[0]["ticker"] == "AAPL"
        assert positions[0]["shares"] == 10

    def test_get_quote(self):
        quote = self.broker.get_quote("AAPL")
        assert quote["price"] == 150.0
        assert quote["ticker"] == "AAPL"

    def test_cancel_pending_order(self):
        # Create a limit order that won't fill
        self.broker.set_prices({"AAPL": 150.0})
        order = self.broker.submit_order("AAPL", "buy", 10, "limit", 100.0)
        assert order.status == OrderStatus.PENDING

        result = self.broker.cancel_order(order.order_id)
        assert result is True
        assert self.broker.get_order_status(order.order_id).status == OrderStatus.CANCELLED

    def test_multiple_buys_average_cost(self):
        self.broker.submit_order("AAPL", "buy", 10)
        self.broker.set_prices({"AAPL": 200.0})
        self.broker.submit_order("AAPL", "buy", 10)

        pos = self.broker.positions["AAPL"]
        assert pos["shares"] == 20
        assert pos["avg_cost"] == 175.0  # (1500 + 2000) / 20
