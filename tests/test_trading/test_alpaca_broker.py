"""Tests for Alpaca broker integration (mocked)."""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.trading.alpaca_broker import AlpacaBroker, _STATUS_MAP
from src.trading.broker_api import OrderStatus


class TestAlpacaBrokerStatusMapping:
    """Test Alpaca status string to OrderStatus mapping."""

    def test_filled_maps_correctly(self):
        assert _STATUS_MAP["filled"] == OrderStatus.FILLED

    def test_new_maps_to_pending(self):
        assert _STATUS_MAP["new"] == OrderStatus.PENDING

    def test_canceled_maps_correctly(self):
        assert _STATUS_MAP["canceled"] == OrderStatus.CANCELLED

    def test_rejected_maps_correctly(self):
        assert _STATUS_MAP["rejected"] == OrderStatus.REJECTED


class TestAlpacaBrokerInit:
    def test_default_paper_mode(self):
        broker = AlpacaBroker(api_key="test", api_secret="test")
        assert broker.paper is True

    def test_base_url_default(self):
        broker = AlpacaBroker(api_key="test", api_secret="test")
        assert "paper" in broker.base_url


class TestAlpacaBrokerSubmitOrder:
    @patch("src.trading.alpaca_broker.AlpacaBroker._get_trading_client")
    def test_submit_market_order(self, mock_client_method):
        mock_client = MagicMock()
        mock_order = MagicMock()
        mock_order.id = "order-123"
        mock_order.symbol = "AAPL"
        mock_order.side = "buy"
        mock_order.qty = 10
        mock_order.order_type = "market"
        mock_order.limit_price = None
        mock_order.status = "filled"
        mock_order.filled_avg_price = 150.0
        mock_order.filled_at = datetime.now()
        mock_order.created_at = datetime.now()
        mock_client.submit_order.return_value = mock_order
        mock_client_method.return_value = mock_client

        broker = AlpacaBroker(api_key="test", api_secret="test")
        order = broker.submit_order("AAPL", "buy", 10)

        assert order.order_id == "order-123"
        assert order.status == OrderStatus.FILLED
        assert order.filled_price == 150.0

    @patch("src.trading.alpaca_broker.AlpacaBroker._get_trading_client")
    def test_submit_order_error(self, mock_client_method):
        mock_client = MagicMock()
        mock_client.submit_order.side_effect = Exception("Connection failed")
        mock_client_method.return_value = mock_client

        broker = AlpacaBroker(api_key="test", api_secret="test")
        order = broker.submit_order("AAPL", "buy", 10)

        assert order.status == OrderStatus.REJECTED
