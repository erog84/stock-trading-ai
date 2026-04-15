"""Broker API routes."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel as PydanticModel
from typing import Optional

from src.trading.executor import TradeExecutor
from src.trading import Signal, Trade
from src.utils.config import config
from datetime import datetime

router = APIRouter()

_executor = TradeExecutor()


class ManualTradeRequest(PydanticModel):
    ticker: str
    side: str  # "buy" or "sell"
    quantity: int
    confidence: float = 1.0
    reason: str = "Manual trade"


@router.get("/status")
def broker_status():
    """Get broker connection status and account info."""
    try:
        account = _executor.broker.get_account_info()
        return {
            "connected": True,
            "broker_type": config.broker_type,
            **account,
        }
    except Exception as e:
        return {
            "connected": False,
            "broker_type": config.broker_type,
            "error": str(e),
        }


@router.get("/positions")
def broker_positions():
    """Get live positions from broker."""
    try:
        return _executor.broker.get_positions()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orders")
def broker_orders(limit: int = 50):
    """Get recent orders."""
    from src.trading.alpaca_broker import AlpacaBroker
    if isinstance(_executor.broker, AlpacaBroker):
        return _executor.broker.get_recent_orders(limit=limit)
    return []


@router.post("/execute")
def execute_trade(request: ManualTradeRequest):
    """Execute a manual trade."""
    signal = Signal.BUY if request.side == "buy" else Signal.SELL

    trade = Trade(
        ticker=request.ticker.upper(),
        signal=signal,
        price=0,  # Will use market price
        quantity=request.quantity,
        timestamp=datetime.now(),
        confidence=request.confidence,
        reason=request.reason,
    )

    # Get current price
    try:
        quote = _executor.broker.get_quote(trade.ticker)
        trade.price = quote["price"]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot get price for {request.ticker}: {e}")

    results = _executor.execute_signals([trade])
    return results[0] if results else {"error": "No result"}


@router.get("/executed-today")
def get_executed_today():
    """Get trades executed today."""
    return _executor.executed_today
