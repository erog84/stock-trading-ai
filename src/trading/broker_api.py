"""Broker API abstraction layer.

Provides a common interface for different brokerage APIs.
Currently supports paper trading (simulation). Real broker
implementations can be added by subclassing BrokerAPI.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from src.trading import Signal, Trade
from src.utils.logger import logger


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Represents a broker order."""
    order_id: str
    ticker: str
    side: str  # "buy" or "sell"
    quantity: int
    order_type: str  # "market", "limit"
    limit_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    filled_at: Optional[datetime] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class BrokerAPI(ABC):
    """Abstract broker API interface."""

    @abstractmethod
    def submit_order(
        self,
        ticker: str,
        side: str,
        quantity: int,
        order_type: str = "market",
        limit_price: Optional[float] = None,
    ) -> Order:
        """Submit an order to the broker."""
        ...

    @abstractmethod
    def get_order_status(self, order_id: str) -> Order:
        """Get the current status of an order."""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        ...

    @abstractmethod
    def get_account_info(self) -> dict:
        """Get account balance and details."""
        ...

    @abstractmethod
    def get_positions(self) -> list[dict]:
        """Get current positions."""
        ...

    @abstractmethod
    def get_quote(self, ticker: str) -> dict:
        """Get current quote for a ticker."""
        ...


class PaperBroker(BrokerAPI):
    """Paper trading broker for simulation and testing."""

    def __init__(self, initial_cash: float = 100_000.0):
        self.cash = initial_cash
        self.positions: dict[str, dict] = {}
        self.orders: dict[str, Order] = {}
        self._order_counter = 0
        self._prices: dict[str, float] = {}

    def set_prices(self, prices: dict[str, float]) -> None:
        """Update simulated market prices."""
        self._prices = prices

    def submit_order(
        self,
        ticker: str,
        side: str,
        quantity: int,
        order_type: str = "market",
        limit_price: Optional[float] = None,
    ) -> Order:
        """Submit and immediately fill a market order (paper trading)."""
        self._order_counter += 1
        order_id = f"PAPER-{self._order_counter:06d}"

        price = self._prices.get(ticker, limit_price)
        if price is None:
            order = Order(
                order_id=order_id, ticker=ticker, side=side,
                quantity=quantity, order_type=order_type,
                limit_price=limit_price, status=OrderStatus.REJECTED,
            )
            self.orders[order_id] = order
            logger.warning(f"Order rejected: no price for {ticker}")
            return order

        # Check limits for limit orders
        if order_type == "limit" and limit_price is not None:
            if side == "buy" and price > limit_price:
                order = Order(
                    order_id=order_id, ticker=ticker, side=side,
                    quantity=quantity, order_type=order_type,
                    limit_price=limit_price, status=OrderStatus.PENDING,
                )
                self.orders[order_id] = order
                return order
            if side == "sell" and price < limit_price:
                order = Order(
                    order_id=order_id, ticker=ticker, side=side,
                    quantity=quantity, order_type=order_type,
                    limit_price=limit_price, status=OrderStatus.PENDING,
                )
                self.orders[order_id] = order
                return order

        # Execute the order
        if side == "buy":
            cost = price * quantity
            if cost > self.cash:
                order = Order(
                    order_id=order_id, ticker=ticker, side=side,
                    quantity=quantity, order_type=order_type,
                    status=OrderStatus.REJECTED,
                )
                self.orders[order_id] = order
                logger.warning(f"Order rejected: insufficient funds (${cost:.2f} > ${self.cash:.2f})")
                return order

            self.cash -= cost
            if ticker in self.positions:
                pos = self.positions[ticker]
                total_shares = pos["shares"] + quantity
                pos["avg_cost"] = (pos["shares"] * pos["avg_cost"] + cost) / total_shares
                pos["shares"] = total_shares
            else:
                self.positions[ticker] = {"shares": quantity, "avg_cost": price}

        elif side == "sell":
            if ticker not in self.positions or self.positions[ticker]["shares"] < quantity:
                order = Order(
                    order_id=order_id, ticker=ticker, side=side,
                    quantity=quantity, order_type=order_type,
                    status=OrderStatus.REJECTED,
                )
                self.orders[order_id] = order
                logger.warning(f"Order rejected: insufficient shares of {ticker}")
                return order

            self.cash += price * quantity
            self.positions[ticker]["shares"] -= quantity
            if self.positions[ticker]["shares"] == 0:
                del self.positions[ticker]

        order = Order(
            order_id=order_id, ticker=ticker, side=side,
            quantity=quantity, order_type=order_type,
            limit_price=limit_price, status=OrderStatus.FILLED,
            filled_price=price, filled_at=datetime.now(),
        )
        self.orders[order_id] = order
        logger.info(f"Paper order filled: {side.upper()} {quantity} {ticker} @ ${price:.2f}")
        return order

    def get_order_status(self, order_id: str) -> Order:
        if order_id not in self.orders:
            raise ValueError(f"Unknown order: {order_id}")
        return self.orders[order_id]

    def cancel_order(self, order_id: str) -> bool:
        if order_id not in self.orders:
            return False
        order = self.orders[order_id]
        if order.status == OrderStatus.PENDING:
            order.status = OrderStatus.CANCELLED
            return True
        return False

    def get_account_info(self) -> dict:
        positions_value = sum(
            pos["shares"] * self._prices.get(ticker, pos["avg_cost"])
            for ticker, pos in self.positions.items()
        )
        return {
            "cash": self.cash,
            "positions_value": positions_value,
            "total_value": self.cash + positions_value,
            "n_positions": len(self.positions),
        }

    def get_positions(self) -> list[dict]:
        return [
            {
                "ticker": ticker,
                "shares": pos["shares"],
                "avg_cost": pos["avg_cost"],
                "current_price": self._prices.get(ticker, pos["avg_cost"]),
                "market_value": pos["shares"] * self._prices.get(ticker, pos["avg_cost"]),
                "unrealized_pnl": pos["shares"] * (self._prices.get(ticker, pos["avg_cost"]) - pos["avg_cost"]),
            }
            for ticker, pos in self.positions.items()
        ]

    def get_quote(self, ticker: str) -> dict:
        price = self._prices.get(ticker)
        if price is None:
            raise ValueError(f"No price available for {ticker}")
        return {"ticker": ticker, "price": price, "timestamp": datetime.now()}
