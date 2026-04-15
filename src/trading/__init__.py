"""Trading engine module."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Signal(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class Trade:
    ticker: str
    signal: Signal
    price: float
    quantity: int
    timestamp: datetime
    confidence: float
    reason: str
