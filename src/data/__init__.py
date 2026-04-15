"""Data ingestion and processing module."""

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


class DataFetcher(ABC):
    """Base interface for all data source fetchers."""

    @abstractmethod
    def fetch(self, tickers: list[str], start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch data for given tickers and date range.

        Returns DataFrame with DatetimeIndex and columns:
        open, high, low, close, volume, ticker
        """
        ...

    @abstractmethod
    def get_source_name(self) -> str:
        """Return the name of this data source."""
        ...
