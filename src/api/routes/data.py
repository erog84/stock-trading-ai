"""Data API routes."""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from src.data.yahoo_finance import YahooFinanceFetcher
from src.data.congress_trades import CongressTradeFetcher

router = APIRouter()

_yahoo = YahooFinanceFetcher()
_congress = CongressTradeFetcher()


@router.get("/ohlcv")
def get_ohlcv(
    tickers: str = Query(..., description="Comma-separated ticker symbols"),
    start_date: str = Query(..., description="Start date YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
):
    """Fetch OHLCV data for given tickers."""
    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    df = _yahoo.fetch(ticker_list, start_date, end_date)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found")
    df = df.reset_index()
    df["date"] = df["date"].astype(str)
    return df.to_dict(orient="records")


@router.get("/stock-info/{ticker}")
def get_stock_info(ticker: str):
    """Get fundamental info for a ticker."""
    info = _yahoo.fetch_info(ticker.upper())
    if not info or info.get("name") is None:
        raise HTTPException(status_code=404, detail=f"No info for {ticker}")
    return info


@router.get("/congress/signals")
def get_congress_signals(
    min_trades: int = Query(3, description="Minimum number of congress members buying"),
    days: int = Query(30, description="Lookback period in days"),
):
    """Get congressional insider trading buy signals."""
    signals = _congress.get_buy_signals(min_trades=min_trades, days=days)
    if signals.empty:
        return []
    return signals.to_dict(orient="records")


@router.get("/congress/top-traded")
def get_congress_top_traded(
    n: int = Query(20, description="Number of tickers to return"),
    days: int = Query(90, description="Lookback period"),
):
    """Get most traded tickers by congress members."""
    return _congress.get_top_traded_tickers(n=n, days=days)
