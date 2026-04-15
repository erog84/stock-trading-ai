"""Congressional insider trading data fetcher.

Uses publicly available congressional trading disclosure data.
Primary source: House/Senate financial disclosure reports.
"""

from typing import Optional
from dataclasses import dataclass
from datetime import datetime
import requests
import pandas as pd

from src.data import DataFetcher
from src.utils.logger import logger

# QuiverQuant-style API for congressional trades
QUIVER_BASE_URL = "https://api.quiverquant.com/beta"


@dataclass
class CongressTrade:
    """Represents a single congressional trade disclosure."""
    politician: str
    party: str
    chamber: str  # House or Senate
    ticker: str
    transaction_type: str  # Purchase, Sale, Exchange
    amount_low: float
    amount_high: float
    disclosure_date: datetime
    transaction_date: Optional[datetime]


class CongressTradeFetcher(DataFetcher):
    """Fetches congressional trading disclosure data."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._cache: Optional[pd.DataFrame] = None

    def get_source_name(self) -> str:
        return "congress_trades"

    def fetch(
        self,
        tickers: list[str],
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch congressional trades for given tickers.

        Returns DataFrame with columns:
        politician, party, chamber, ticker, transaction_type,
        amount_low, amount_high, disclosure_date, transaction_date
        """
        all_trades = self._fetch_all_trades()
        if all_trades.empty:
            return all_trades

        # Filter by tickers
        filtered = all_trades[all_trades["ticker"].isin(tickers)]

        # Filter by date range
        filtered = filtered[filtered["disclosure_date"] >= start_date]
        if end_date:
            filtered = filtered[filtered["disclosure_date"] <= end_date]

        logger.info(f"Found {len(filtered)} congressional trades for {tickers}")
        return filtered

    def _fetch_all_trades(self) -> pd.DataFrame:
        """Fetch all available congressional trades (cached)."""
        if self._cache is not None:
            return self._cache

        try:
            # Try QuiverQuant API if key available
            if self.api_key:
                return self._fetch_from_quiver()

            # Fallback: fetch from public House/Senate disclosure data
            return self._fetch_from_public_disclosures()

        except Exception as e:
            logger.error(f"Error fetching congressional trades: {e}")
            return pd.DataFrame()

    def _fetch_from_quiver(self) -> pd.DataFrame:
        """Fetch from QuiverQuant API."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            resp = requests.get(
                f"{QUIVER_BASE_URL}/historical/congresstrading",
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            df = pd.DataFrame(data)
            df = self._normalize_quiver_data(df)
            self._cache = df
            return df

        except requests.RequestException as e:
            logger.error(f"QuiverQuant API error: {e}")
            return pd.DataFrame()

    def _normalize_quiver_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize QuiverQuant data to standard schema."""
        if df.empty:
            return df

        column_map = {
            "Representative": "politician",
            "Party": "party",
            "House": "chamber",
            "Ticker": "ticker",
            "Transaction": "transaction_type",
            "Amount": "amount_range",
            "ReportDate": "disclosure_date",
            "TransactionDate": "transaction_date",
        }

        # Only rename columns that exist
        rename = {k: v for k, v in column_map.items() if k in df.columns}
        df = df.rename(columns=rename)

        if "disclosure_date" in df.columns:
            df["disclosure_date"] = pd.to_datetime(df["disclosure_date"])
        if "transaction_date" in df.columns:
            df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")

        # Parse amount ranges like "$1,001 - $15,000"
        if "amount_range" in df.columns:
            amounts = df["amount_range"].str.replace(r"[\$,]", "", regex=True).str.split(" - ", expand=True)
            df["amount_low"] = pd.to_numeric(amounts[0], errors="coerce").fillna(0)
            df["amount_high"] = pd.to_numeric(amounts[1] if 1 in amounts.columns else amounts[0], errors="coerce").fillna(0)
            df = df.drop(columns=["amount_range"])

        return df

    def _fetch_from_public_disclosures(self) -> pd.DataFrame:
        """Fetch from free public sources for congressional trades.

        Uses the House Stock Watcher API (free, community-maintained)
        and Senate disclosures as fallback.
        """
        df = self._fetch_house_stock_watcher()
        if not df.empty:
            self._cache = df
            return df

        logger.warning(
            "Could not fetch congressional trades from free sources. "
            "Set QUIVER_API_KEY in .env for full data."
        )
        return pd.DataFrame(columns=[
            "politician", "party", "chamber", "ticker",
            "transaction_type", "amount_low", "amount_high",
            "disclosure_date", "transaction_date",
        ])

    def _fetch_house_stock_watcher(self) -> pd.DataFrame:
        """Fetch from House Stock Watcher (free, no key)."""
        try:
            # Community-maintained API for House financial disclosures
            resp = requests.get(
                "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json",
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)

            # Normalize column names
            column_map = {
                "representative": "politician",
                "party": "party",
                "type": "transaction_type",
                "ticker": "ticker",
                "amount": "amount_range",
                "disclosure_date": "disclosure_date",
                "transaction_date": "transaction_date",
            }
            rename = {k: v for k, v in column_map.items() if k in df.columns}
            df = df.rename(columns=rename)
            df["chamber"] = "House"

            # Parse dates
            if "disclosure_date" in df.columns:
                df["disclosure_date"] = pd.to_datetime(df["disclosure_date"], errors="coerce")
            if "transaction_date" in df.columns:
                df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")

            # Parse amount ranges
            if "amount_range" in df.columns:
                amounts = df["amount_range"].str.replace(r"[\$,]", "", regex=True).str.split(" - ", expand=True)
                df["amount_low"] = pd.to_numeric(amounts[0], errors="coerce").fillna(0)
                df["amount_high"] = pd.to_numeric(amounts.get(1, amounts[0]), errors="coerce").fillna(0)
                df = df.drop(columns=["amount_range"], errors="ignore")

            # Filter to stock transactions only (exclude crypto, options, etc.)
            if "ticker" in df.columns:
                df = df[df["ticker"].str.match(r"^[A-Z]{1,5}$", na=False)]

            logger.info(f"Fetched {len(df)} House congressional trades (free source)")
            return df

        except Exception as e:
            logger.warning(f"House Stock Watcher fetch failed: {e}")
            return pd.DataFrame()

    def get_top_traded_tickers(self, n: int = 20, days: int = 90) -> list[str]:
        """Get the most frequently traded tickers by congress members."""
        trades = self._fetch_all_trades()
        if trades.empty:
            return []

        cutoff = (datetime.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
        recent = trades[trades["disclosure_date"] >= cutoff]
        return recent["ticker"].value_counts().head(n).index.tolist()

    def get_buy_signals(self, min_trades: int = 3, days: int = 30) -> pd.DataFrame:
        """Identify tickers with unusual buying activity from congress.

        A signal is generated when multiple congress members buy the same
        stock within a short time window.
        """
        trades = self._fetch_all_trades()
        if trades.empty:
            return pd.DataFrame()

        cutoff = (datetime.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
        recent_buys = trades[
            (trades["disclosure_date"] >= cutoff)
            & (trades["transaction_type"].str.lower().str.contains("purchase", na=False))
        ]

        # Count unique politicians buying each ticker
        signals = (
            recent_buys.groupby("ticker")
            .agg(
                buy_count=("politician", "nunique"),
                total_trades=("ticker", "count"),
                politicians=("politician", lambda x: ", ".join(x.unique()[:5])),
                avg_amount=("amount_low", "mean"),
            )
            .reset_index()
        )

        signals = signals[signals["buy_count"] >= min_trades].sort_values(
            "buy_count", ascending=False
        )

        return signals
