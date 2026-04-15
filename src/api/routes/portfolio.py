"""Portfolio API routes."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel as PydanticModel
from typing import Optional

from src.trading.portfolio import Portfolio
from src.trading.broker_api import PaperBroker

router = APIRouter()

# In-memory portfolio and broker (will be replaced with persistent storage)
_portfolio = Portfolio(initial_cash=100_000.0)
_broker = PaperBroker(initial_cash=100_000.0)


class TradeRequest(PydanticModel):
    ticker: str
    side: str  # "buy" or "sell"
    quantity: int
    order_type: str = "market"
    limit_price: Optional[float] = None


@router.get("/summary")
def get_portfolio_summary():
    """Get portfolio performance summary."""
    return _portfolio.get_performance_summary()


@router.get("/positions")
def get_positions():
    """Get current positions."""
    return [
        {
            "ticker": ticker,
            "shares": pos.shares,
            "avg_cost": pos.avg_cost,
            "current_price": pos.current_price,
            "market_value": pos.market_value,
            "unrealized_pnl": pos.unrealized_pnl,
            "unrealized_pnl_pct": pos.unrealized_pnl_pct,
        }
        for ticker, pos in _portfolio.positions.items()
    ]


@router.get("/trades")
def get_trade_history():
    """Get trade history."""
    df = _portfolio.get_trade_history_df()
    if df.empty:
        return []
    return df.to_dict(orient="records")


@router.get("/daily-values")
def get_daily_values():
    """Get daily portfolio values for charting."""
    return _portfolio.daily_values


@router.post("/trade")
def submit_trade(request: TradeRequest):
    """Submit a manual trade through paper broker."""
    order = _broker.submit_order(
        ticker=request.ticker,
        side=request.side,
        quantity=request.quantity,
        order_type=request.order_type,
        limit_price=request.limit_price,
    )
    return {
        "order_id": order.order_id,
        "status": order.status.value,
        "filled_price": order.filled_price,
    }


@router.get("/account")
def get_account_info():
    """Get broker account info."""
    return _broker.get_account_info()
