"""Alpha Vantage data fetcher for technical indicators and sector data."""

from typing import Optional
import time
import requests
import pandas as pd

from src.data import DataFetcher
from src.utils.config import config
from src.utils.logger import logger

BASE_URL = "https://www.alphavantage.co/query"

# Free tier: 25 requests/day
RATE_LIMIT_DELAY = 12  # seconds between requests


class AlphaVantageFetcher(DataFetcher):
    """Fetches technical indicators and sector performance from Alpha Vantage."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.alpha_vantage_api_key
        if not self.api_key:
            logger.warning("Alpha Vantage API key not set. Set ALPHA_VANTAGE_API_KEY in .env")

    def get_source_name(self) -> str:
        return "alpha_vantage"

    def _request(self, params: dict) -> dict:
        """Make a rate-limited request to Alpha Vantage API."""
        params["apikey"] = self.api_key
        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if "Error Message" in data:
                raise ValueError(data["Error Message"])
            if "Note" in data:
                logger.warning(f"Alpha Vantage rate limit note: {data['Note']}")

            time.sleep(RATE_LIMIT_DELAY)
            return data
        except requests.RequestException as e:
            logger.error(f"Alpha Vantage request error: {e}")
            raise

    def fetch(
        self,
        tickers: list[str],
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch daily adjusted data from Alpha Vantage."""
        frames = []
        for ticker in tickers:
            try:
                data = self._request({
                    "function": "TIME_SERIES_DAILY",
                    "symbol": ticker,
                    "outputsize": "full",
                })

                ts_key = "Time Series (Daily)"
                if ts_key not in data:
                    logger.warning(f"No time series data for {ticker}")
                    continue

                df = pd.DataFrame.from_dict(data[ts_key], orient="index")
                df.index = pd.to_datetime(df.index)
                df.index.name = "date"
                df = df.sort_index()

                df = df.rename(columns={
                    "1. open": "open",
                    "2. high": "high",
                    "3. low": "low",
                    "4. close": "close",
                    "5. volume": "volume",
                })
                df = df[["open", "high", "low", "close", "volume"]].astype(float)
                df["ticker"] = ticker

                # Filter date range
                df = df[df.index >= start_date]
                if end_date:
                    df = df[df.index <= end_date]

                frames.append(df)
                logger.debug(f"Fetched {len(df)} rows for {ticker} from Alpha Vantage")

            except Exception as e:
                logger.error(f"Error fetching {ticker} from Alpha Vantage: {e}")

        if not frames:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "ticker"])

        return pd.concat(frames)

    def fetch_rsi(self, ticker: str, period: int = 14) -> pd.Series:
        """Fetch RSI (Relative Strength Index) for a ticker."""
        data = self._request({
            "function": "RSI",
            "symbol": ticker,
            "interval": "daily",
            "time_period": str(period),
            "series_type": "close",
        })

        key = "Technical Analysis: RSI"
        if key not in data:
            return pd.Series(dtype=float, name="rsi")

        df = pd.DataFrame.from_dict(data[key], orient="index")
        df.index = pd.to_datetime(df.index)
        series = df["RSI"].astype(float).sort_index()
        series.name = "rsi"
        return series

    def fetch_macd(self, ticker: str) -> pd.DataFrame:
        """Fetch MACD for a ticker."""
        data = self._request({
            "function": "MACD",
            "symbol": ticker,
            "interval": "daily",
            "series_type": "close",
        })

        key = "Technical Analysis: MACD"
        if key not in data:
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(data[key], orient="index").astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df.columns = ["macd", "macd_signal", "macd_hist"]
        return df

    def fetch_sector_performance(self) -> dict:
        """Fetch real-time sector performance."""
        data = self._request({"function": "SECTOR"})
        return data
