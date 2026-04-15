"""Alpaca brokerage API integration.

Supports both paper and live trading via alpaca-py SDK.
Paper trading is free with no account minimum.
"""

from datetime import datetime
from typing import Optional

from src.trading.broker_api import BrokerAPI, Order, OrderStatus
from src.utils.config import config
from src.utils.logger import logger

# Map Alpaca order statuses to our OrderStatus enum
_STATUS_MAP = {
    "new": OrderStatus.PENDING,
    "accepted": OrderStatus.PENDING,
    "pending_new": OrderStatus.PENDING,
    "accepted_for_bidding": OrderStatus.PENDING,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELLED,
    "cancelled": OrderStatus.CANCELLED,
    "expired": OrderStatus.CANCELLED,
    "rejected": OrderStatus.REJECTED,
    "stopped": OrderStatus.CANCELLED,
    "suspended": OrderStatus.PENDING,
}


class AlpacaBroker(BrokerAPI):
    """Alpaca brokerage API implementation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: Optional[str] = None,
        paper: bool = True,
    ):
        self.api_key = api_key or config.alpaca_api_key
        self.api_secret = api_secret or config.alpaca_api_secret
        self.base_url = base_url or config.alpaca_base_url
        self.paper = paper
        self._client = None
        self._trading_client = None

    def _get_trading_client(self):
        """Lazy-init the Alpaca trading client."""
        if self._trading_client is None:
            try:
                from alpaca.trading.client import TradingClient
                self._trading_client = TradingClient(
                    api_key=self.api_key,
                    secret_key=self.api_secret,
                    paper=self.paper,
                )
            except ImportError:
                raise ImportError("alpaca-py not installed. Run: pip install alpaca-py")
        return self._trading_client

    def submit_order(
        self,
        ticker: str,
        side: str,
        quantity: int,
        order_type: str = "market",
        limit_price: Optional[float] = None,
    ) -> Order:
        """Submit an order to Alpaca."""
        from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        client = self._get_trading_client()
        alpaca_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

        try:
            if order_type == "limit" and limit_price is not None:
                request = LimitOrderRequest(
                    symbol=ticker,
                    qty=quantity,
                    side=alpaca_side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price,
                )
            else:
                request = MarketOrderRequest(
                    symbol=ticker,
                    qty=quantity,
                    side=alpaca_side,
                    time_in_force=TimeInForce.DAY,
                )

            alpaca_order = client.submit_order(request)
            order = self._convert_order(alpaca_order)
            logger.info(f"Alpaca order submitted: {side.upper()} {quantity} {ticker} -> {order.order_id}")
            return order

        except Exception as e:
            logger.error(f"Alpaca order error: {e}")
            return Order(
                order_id=f"ERROR-{datetime.now().timestamp():.0f}",
                ticker=ticker,
                side=side,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                status=OrderStatus.REJECTED,
            )

    def get_order_status(self, order_id: str) -> Order:
        """Get order status from Alpaca."""
        client = self._get_trading_client()
        try:
            alpaca_order = client.get_order_by_id(order_id)
            return self._convert_order(alpaca_order)
        except Exception as e:
            logger.error(f"Error getting order {order_id}: {e}")
            raise

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        client = self._get_trading_client()
        try:
            client.cancel_order_by_id(order_id)
            logger.info(f"Order {order_id} cancelled")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    def get_account_info(self) -> dict:
        """Get Alpaca account info."""
        client = self._get_trading_client()
        try:
            account = client.get_account()
            return {
                "cash": float(account.cash),
                "positions_value": float(account.long_market_value),
                "total_value": float(account.equity),
                "buying_power": float(account.buying_power),
                "n_positions": int(account.position_qty) if hasattr(account, 'position_qty') else 0,
                "day_trade_count": int(account.daytrade_count),
                "account_status": str(account.status),
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            raise

    def get_positions(self) -> list[dict]:
        """Get current positions from Alpaca."""
        client = self._get_trading_client()
        try:
            positions = client.get_all_positions()
            return [
                {
                    "ticker": pos.symbol,
                    "shares": int(pos.qty),
                    "avg_cost": float(pos.avg_entry_price),
                    "current_price": float(pos.current_price),
                    "market_value": float(pos.market_value),
                    "unrealized_pnl": float(pos.unrealized_pl),
                    "unrealized_pnl_pct": float(pos.unrealized_plpc) * 100,
                }
                for pos in positions
            ]
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            raise

    def get_quote(self, ticker: str) -> dict:
        """Get latest quote from Alpaca."""
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestQuoteRequest

            data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.api_secret,
            )
            request = StockLatestQuoteRequest(symbol_or_symbols=ticker)
            quote = data_client.get_stock_latest_quote(request)

            if ticker in quote:
                q = quote[ticker]
                return {
                    "ticker": ticker,
                    "price": float(q.ask_price + q.bid_price) / 2,
                    "bid": float(q.bid_price),
                    "ask": float(q.ask_price),
                    "timestamp": datetime.now(),
                }
            raise ValueError(f"No quote for {ticker}")

        except ImportError:
            raise ImportError("alpaca-py not installed. Run: pip install alpaca-py")
        except Exception as e:
            logger.error(f"Error getting quote for {ticker}: {e}")
            raise

    def get_recent_orders(self, limit: int = 50) -> list[dict]:
        """Get recent orders from Alpaca."""
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        client = self._get_trading_client()
        try:
            request = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit)
            orders = client.get_orders(request)
            return [
                {
                    "order_id": str(order.id),
                    "ticker": order.symbol,
                    "side": str(order.side),
                    "quantity": int(order.qty),
                    "order_type": str(order.order_type),
                    "status": str(order.status),
                    "filled_price": float(order.filled_avg_price) if order.filled_avg_price else None,
                    "created_at": str(order.created_at),
                }
                for order in orders
            ]
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return []

    def _convert_order(self, alpaca_order) -> Order:
        """Convert Alpaca order object to our Order model."""
        status_str = str(alpaca_order.status).lower()
        status = _STATUS_MAP.get(status_str, OrderStatus.PENDING)

        return Order(
            order_id=str(alpaca_order.id),
            ticker=alpaca_order.symbol,
            side=str(alpaca_order.side).lower(),
            quantity=int(alpaca_order.qty),
            order_type=str(alpaca_order.order_type).lower(),
            limit_price=float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
            status=status,
            filled_price=float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else None,
            filled_at=alpaca_order.filled_at,
            created_at=alpaca_order.created_at,
        )
