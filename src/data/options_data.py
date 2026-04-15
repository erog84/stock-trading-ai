"""Options chain data and unusual activity detection via yfinance.

Free, no API key required.
"""

from typing import Optional
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf

from src.data import DataFetcher
from src.utils.logger import logger


class OptionsDataFetcher(DataFetcher):
    """Fetches options chain data and computes derived features."""

    def get_source_name(self) -> str:
        return "options"

    def fetch(
        self,
        tickers: list[str],
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch options-derived features for given tickers.

        Returns DataFrame with columns:
        ticker, put_call_ratio, iv_avg, iv_skew, options_volume, unusual_volume
        """
        features = []

        for ticker in tickers:
            try:
                data = self._analyze_options(ticker)
                if data:
                    features.append(data)
            except Exception as e:
                logger.debug(f"Options data error for {ticker}: {e}")

        if not features:
            return pd.DataFrame(columns=[
                "ticker", "put_call_ratio", "iv_avg", "iv_skew",
                "options_volume", "unusual_volume",
            ])

        return pd.DataFrame(features)

    def _analyze_options(self, ticker: str) -> Optional[dict]:
        """Analyze options chain for a single ticker."""
        stock = yf.Ticker(ticker)

        try:
            expirations = stock.options
        except Exception:
            return None

        if not expirations:
            return None

        # Use nearest expiration
        nearest_exp = expirations[0]

        try:
            chain = stock.option_chain(nearest_exp)
        except Exception:
            return None

        calls = chain.calls
        puts = chain.puts

        if calls.empty and puts.empty:
            return None

        # Put/Call ratio (volume-based)
        call_volume = calls["volume"].sum() if "volume" in calls.columns else 0
        put_volume = puts["volume"].sum() if "volume" in puts.columns else 0

        if call_volume > 0:
            put_call_ratio = put_volume / call_volume
        else:
            put_call_ratio = 0.0

        # Implied volatility
        call_iv = calls["impliedVolatility"].dropna() if "impliedVolatility" in calls.columns else pd.Series(dtype=float)
        put_iv = puts["impliedVolatility"].dropna() if "impliedVolatility" in puts.columns else pd.Series(dtype=float)
        all_iv = pd.concat([call_iv, put_iv])
        iv_avg = all_iv.mean() if len(all_iv) > 0 else 0.0

        # IV Skew (difference between OTM put IV and OTM call IV)
        # Positive skew = more demand for downside protection
        iv_skew = 0.0
        if len(put_iv) > 0 and len(call_iv) > 0:
            iv_skew = put_iv.mean() - call_iv.mean()

        # Total options volume
        total_volume = (call_volume or 0) + (put_volume or 0)

        # Open interest for unusual activity detection
        call_oi = calls["openInterest"].sum() if "openInterest" in calls.columns else 0
        put_oi = puts["openInterest"].sum() if "openInterest" in puts.columns else 0
        total_oi = (call_oi or 0) + (put_oi or 0)

        # Unusual volume = volume / open interest ratio
        unusual_volume = total_volume / max(total_oi, 1)

        return {
            "ticker": ticker,
            "put_call_ratio": round(put_call_ratio, 3),
            "iv_avg": round(iv_avg, 4),
            "iv_skew": round(iv_skew, 4),
            "options_volume": int(total_volume),
            "unusual_volume": round(unusual_volume, 3),
            "call_volume": int(call_volume or 0),
            "put_volume": int(put_volume or 0),
            "nearest_expiry": nearest_exp,
        }

    def get_unusual_activity(self, tickers: list[str], threshold: float = 2.0) -> pd.DataFrame:
        """Find tickers with unusual options activity.

        Args:
            tickers: List of tickers to check
            threshold: Volume/OI ratio threshold (2.0 = 2x normal)
        """
        data = self.fetch(tickers, start_date="")
        if data.empty:
            return data

        unusual = data[data["unusual_volume"] >= threshold].sort_values(
            "unusual_volume", ascending=False,
        )
        return unusual
