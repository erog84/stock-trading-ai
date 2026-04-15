"""Yahoo Finance data fetcher using yfinance."""

from typing import Optional
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

from src.data import DataFetcher
from src.utils.logger import logger


class YahooFinanceFetcher(DataFetcher):
    """Fetches OHLCV and fundamental data from Yahoo Finance."""

    def get_source_name(self) -> str:
        return "yahoo_finance"

    def fetch(
        self,
        tickers: list[str],
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV data for given tickers.

        Args:
            tickers: List of stock ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (defaults to today)

        Returns:
            DataFrame with DatetimeIndex and columns:
            open, high, low, close, volume, ticker
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Fetching Yahoo Finance data for {len(tickers)} tickers from {start_date} to {end_date}")

        frames = []
        for ticker in tickers:
            try:
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True,
                )
                if data.empty:
                    logger.warning(f"No data returned for {ticker}")
                    continue

                # Flatten MultiIndex columns if present
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                data = data.rename(columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                })
                data["ticker"] = ticker
                data.index.name = "date"
                frames.append(data[["open", "high", "low", "close", "volume", "ticker"]])
                logger.debug(f"Fetched {len(data)} rows for {ticker}")

            except Exception as e:
                logger.error(f"Error fetching {ticker}: {e}")

        if not frames:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "ticker"])

        result = pd.concat(frames)
        logger.info(f"Total rows fetched: {len(result)}")
        return result

    def fetch_info(self, ticker: str) -> dict:
        """Fetch fundamental info for a single ticker."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                "ticker": ticker,
                "name": info.get("longName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", None),
                "forward_pe": info.get("forwardPE", None),
                "dividend_yield": info.get("dividendYield", None),
                "beta": info.get("beta", None),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh", None),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow", None),
            }
        except Exception as e:
            logger.error(f"Error fetching info for {ticker}: {e}")
            return {"ticker": ticker}
