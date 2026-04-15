"""Federal Reserve Economic Data (FRED) fetcher."""

from typing import Optional
import pandas as pd

from src.data import DataFetcher
from src.utils.config import config
from src.utils.logger import logger

# Key economic indicators and their FRED series IDs
ECONOMIC_INDICATORS = {
    "fed_funds_rate": "FEDFUNDS",
    "unemployment_rate": "UNRATE",
    "cpi": "CPIAUCSL",
    "gdp": "GDP",
    "treasury_10y": "DGS10",
    "treasury_2y": "DGS2",
    "yield_spread": None,  # computed: 10y - 2y
    "vix": "VIXCLS",
    "sp500": "SP500",
    "consumer_sentiment": "UMCSENT",
    "industrial_production": "INDPRO",
    "housing_starts": "HOUST",
}


class FredFetcher(DataFetcher):
    """Fetches economic indicators from FRED."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.fred_api_key
        self._fred = None

    def _get_fred(self):
        """Lazy-init FRED client."""
        if self._fred is None:
            try:
                from fredapi import Fred
                self._fred = Fred(api_key=self.api_key)
            except ImportError:
                logger.error("fredapi not installed. Run: pip install fredapi")
                raise
        return self._fred

    def get_source_name(self) -> str:
        return "fred"

    def fetch(
        self,
        tickers: list[str],
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch economic indicators. 'tickers' here are indicator names from ECONOMIC_INDICATORS."""
        fred = self._get_fred()
        frames = []

        for indicator in tickers:
            series_id = ECONOMIC_INDICATORS.get(indicator)
            if series_id is None:
                if indicator == "yield_spread":
                    # Compute yield spread
                    df = self._compute_yield_spread(start_date, end_date)
                    if df is not None:
                        frames.append(df)
                    continue
                logger.warning(f"Unknown indicator: {indicator}")
                continue

            try:
                series = fred.get_series(
                    series_id,
                    observation_start=start_date,
                    observation_end=end_date,
                )
                df = series.to_frame(name=indicator)
                df.index.name = "date"
                frames.append(df)
                logger.debug(f"Fetched {len(df)} rows for {indicator}")

            except Exception as e:
                logger.error(f"Error fetching {indicator} ({series_id}): {e}")

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, axis=1)
        result = result.sort_index()
        return result

    def _compute_yield_spread(
        self, start_date: str, end_date: Optional[str]
    ) -> Optional[pd.DataFrame]:
        """Compute 10Y-2Y Treasury yield spread (recession indicator)."""
        try:
            fred = self._get_fred()
            t10 = fred.get_series("DGS10", observation_start=start_date, observation_end=end_date)
            t2 = fred.get_series("DGS2", observation_start=start_date, observation_end=end_date)

            spread = (t10 - t2).dropna()
            df = spread.to_frame(name="yield_spread")
            df.index.name = "date"
            return df

        except Exception as e:
            logger.error(f"Error computing yield spread: {e}")
            return None

    def fetch_all_indicators(
        self, start_date: str, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch all available economic indicators."""
        return self.fetch(list(ECONOMIC_INDICATORS.keys()), start_date, end_date)
