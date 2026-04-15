"""Tests for Yahoo Finance data fetcher."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from src.data.yahoo_finance import YahooFinanceFetcher


class TestYahooFinanceFetcher:
    def setup_method(self):
        self.fetcher = YahooFinanceFetcher()

    def test_get_source_name(self):
        assert self.fetcher.get_source_name() == "yahoo_finance"

    @patch("src.data.yahoo_finance.yf.download")
    def test_fetch_returns_dataframe(self, mock_download):
        """Test that fetch returns a properly formatted DataFrame."""
        mock_data = pd.DataFrame({
            "Open": [150.0, 151.0],
            "High": [152.0, 153.0],
            "Low": [149.0, 150.0],
            "Close": [151.0, 152.0],
            "Volume": [1000000, 1100000],
        }, index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
        mock_download.return_value = mock_data

        result = self.fetcher.fetch(["AAPL"], "2024-01-01", "2024-01-05")

        assert isinstance(result, pd.DataFrame)
        assert "close" in result.columns
        assert "ticker" in result.columns
        assert result["ticker"].iloc[0] == "AAPL"
        assert len(result) == 2

    @patch("src.data.yahoo_finance.yf.download")
    def test_fetch_empty_ticker(self, mock_download):
        """Test handling of empty data."""
        mock_download.return_value = pd.DataFrame()
        result = self.fetcher.fetch(["INVALID"], "2024-01-01")
        assert result.empty

    @patch("src.data.yahoo_finance.yf.download")
    def test_fetch_multiple_tickers(self, mock_download):
        """Test fetching multiple tickers."""
        mock_data = pd.DataFrame({
            "Open": [150.0],
            "High": [152.0],
            "Low": [149.0],
            "Close": [151.0],
            "Volume": [1000000],
        }, index=pd.to_datetime(["2024-01-02"]))
        mock_download.return_value = mock_data

        result = self.fetcher.fetch(["AAPL", "MSFT"], "2024-01-01")
        # Should be called twice (once per ticker)
        assert mock_download.call_count == 2

    @patch("src.data.yahoo_finance.yf.download")
    def test_fetch_handles_error(self, mock_download):
        """Test error handling during fetch."""
        mock_download.side_effect = Exception("Network error")
        result = self.fetcher.fetch(["AAPL"], "2024-01-01")
        assert result.empty
