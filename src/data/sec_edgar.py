"""SEC EDGAR filing fetcher and sentiment analyzer.

Free, no API key required. Uses sec-edgar-downloader for filing retrieval
and Loughran-McDonald sentiment analysis.
"""

from typing import Optional
from pathlib import Path
from datetime import datetime, timedelta
import re
import pandas as pd

from src.data import DataFetcher
from src.data.sentiment import SentimentScorer
from src.utils.config import config
from src.utils.logger import logger


class SECEdgarFetcher(DataFetcher):
    """Fetches SEC filings and computes sentiment features."""

    def __init__(self, download_dir: Optional[str] = None):
        self.download_dir = Path(download_dir or config.data_dir) / "raw" / "sec_edgar"
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.scorer = SentimentScorer()

    def get_source_name(self) -> str:
        return "sec_edgar"

    def fetch(
        self,
        tickers: list[str],
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch SEC filing sentiment features for given tickers.

        Returns DataFrame with columns:
        ticker, filing_date, filing_type, sentiment_score, word_count
        """
        frames = []

        for ticker in tickers:
            try:
                filings = self._fetch_filings(ticker, start_date, end_date)
                if filings:
                    frames.append(pd.DataFrame(filings))
            except Exception as e:
                logger.warning(f"SEC EDGAR error for {ticker}: {e}")

        if not frames:
            return pd.DataFrame(columns=[
                "ticker", "filing_date", "filing_type",
                "sentiment_score", "word_count",
            ])

        result = pd.concat(frames, ignore_index=True)
        result["filing_date"] = pd.to_datetime(result["filing_date"])
        return result

    def _fetch_filings(
        self,
        ticker: str,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> list[dict]:
        """Fetch and analyze filings for a single ticker."""
        try:
            from sec_edgar_downloader import Downloader
        except ImportError:
            logger.error("sec-edgar-downloader not installed")
            return []

        filings = []
        dl = Downloader("StockTradingAI", "user@example.com", str(self.download_dir))

        for filing_type in ["10-K", "10-Q", "8-K"]:
            try:
                dl.get(filing_type, ticker, after=start_date, before=end_date, limit=5)

                # Find downloaded files
                filing_dir = self.download_dir / "sec-edgar-filings" / ticker / filing_type
                if not filing_dir.exists():
                    continue

                for filing_path in filing_dir.rglob("*.txt"):
                    text = self._extract_text(filing_path)
                    if not text:
                        continue

                    # Get filing date from path or file
                    filing_date = self._extract_date(filing_path, text)

                    sentiment = self.scorer.score_detailed(text)
                    filings.append({
                        "ticker": ticker,
                        "filing_date": filing_date,
                        "filing_type": filing_type,
                        "sentiment_score": sentiment["sentiment_score"],
                        "positive_pct": sentiment["positive_pct"],
                        "negative_pct": sentiment["negative_pct"],
                        "uncertainty_pct": sentiment["uncertainty_pct"],
                        "word_count": sentiment["total_words"],
                    })

            except Exception as e:
                logger.debug(f"Could not fetch {filing_type} for {ticker}: {e}")

        return filings

    def _extract_text(self, path: Path) -> str:
        """Extract plain text from SEC filing."""
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            # Strip HTML tags
            text = re.sub(r"<[^>]+>", " ", content)
            # Strip excessive whitespace
            text = re.sub(r"\s+", " ", text)
            # Limit text length for performance
            return text[:100_000]
        except Exception:
            return ""

    def _extract_date(self, path: Path, text: str) -> str:
        """Extract filing date from path structure or content."""
        # Try to find date in path (common format: .../0001234567-24-001234/...)
        parts = str(path).split("/") + str(path).split("\\")
        for part in parts:
            match = re.match(r"(\d{4})-(\d{2})-(\d{2})", part)
            if match:
                return match.group(0)

        # Try to find FILED date in text
        match = re.search(r"FILED\s+AS\s+OF\s+DATE:\s+(\d{8})", text)
        if match:
            d = match.group(1)
            return f"{d[:4]}-{d[4:6]}-{d[6:8]}"

        return datetime.now().strftime("%Y-%m-%d")

    def get_sentiment_features(
        self,
        tickers: list[str],
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get aggregated sentiment features per ticker per date.

        Returns DataFrame indexed by date with columns per ticker:
        sec_sentiment, sec_filing_count_30d, sec_latest_filing_type
        """
        filings = self.fetch(tickers, start_date, end_date)
        if filings.empty:
            return pd.DataFrame()

        # Aggregate per ticker
        features = []
        for ticker in tickers:
            ticker_filings = filings[filings["ticker"] == ticker].sort_values("filing_date")
            if ticker_filings.empty:
                continue

            # Rolling sentiment
            latest = ticker_filings.iloc[-1]
            features.append({
                "ticker": ticker,
                "sec_sentiment": latest["sentiment_score"],
                "sec_filing_count_30d": len(ticker_filings[
                    ticker_filings["filing_date"] >= (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
                ]),
                "sec_latest_filing_type": latest["filing_type"],
                "sec_uncertainty_pct": latest["uncertainty_pct"],
            })

        return pd.DataFrame(features) if features else pd.DataFrame()
