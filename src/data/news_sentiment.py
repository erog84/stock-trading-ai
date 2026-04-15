"""News sentiment via free RSS feeds.

No API key required. Sources: Yahoo Finance, MarketWatch, Google News.
"""

from typing import Optional
from datetime import datetime, timedelta
import re
import pandas as pd

from src.data import DataFetcher
from src.data.sentiment import score_text
from src.utils.logger import logger

# RSS feed URLs (free, no auth)
RSS_FEEDS = {
    "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
    "marketwatch": "https://feeds.marketwatch.com/marketwatch/marketpulse/",
    "google_news_business": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB",
}


class NewsSentimentFetcher(DataFetcher):
    """Fetches news headlines and scores sentiment."""

    def get_source_name(self) -> str:
        return "news_sentiment"

    def fetch(
        self,
        tickers: list[str],
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch news articles mentioning given tickers and score sentiment."""
        try:
            import feedparser
        except ImportError:
            logger.error("feedparser not installed. Run: pip install feedparser")
            return pd.DataFrame()

        all_articles = []

        for feed_name, feed_url in RSS_FEEDS.items():
            try:
                feed = feedparser.parse(feed_url)

                for entry in feed.entries:
                    title = entry.get("title", "")
                    summary = entry.get("summary", "")
                    published = entry.get("published_parsed") or entry.get("updated_parsed")

                    if published:
                        pub_date = datetime(*published[:6])
                    else:
                        pub_date = datetime.now()

                    # Check if article mentions any of our tickers
                    text = f"{title} {summary}".upper()
                    mentioned_tickers = [t for t in tickers if t in text]

                    if mentioned_tickers:
                        sentiment = score_text(f"{title} {summary}")
                        for ticker in mentioned_tickers:
                            all_articles.append({
                                "ticker": ticker,
                                "date": pub_date.strftime("%Y-%m-%d"),
                                "source": feed_name,
                                "title": title[:200],
                                "sentiment": sentiment,
                            })

            except Exception as e:
                logger.warning(f"Error fetching feed {feed_name}: {e}")

        if not all_articles:
            return pd.DataFrame(columns=["ticker", "date", "source", "title", "sentiment"])

        df = pd.DataFrame(all_articles)
        df["date"] = pd.to_datetime(df["date"])

        # Filter date range
        df = df[df["date"] >= start_date]
        if end_date:
            df = df[df["date"] <= end_date]

        return df

    def get_sentiment_features(
        self,
        tickers: list[str],
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get aggregated news sentiment features per ticker.

        Returns DataFrame with columns:
        news_sentiment_1d, news_sentiment_7d, news_volume_1d, news_volume_7d
        """
        articles = self.fetch(tickers, start_date, end_date)
        if articles.empty:
            return pd.DataFrame()

        features = []
        now = datetime.now()

        for ticker in tickers:
            ticker_articles = articles[articles["ticker"] == ticker]
            if ticker_articles.empty:
                features.append({
                    "ticker": ticker,
                    "news_sentiment_1d": 0.0,
                    "news_sentiment_7d": 0.0,
                    "news_volume_1d": 0,
                    "news_volume_7d": 0,
                })
                continue

            # Last 1 day
            cutoff_1d = (now - timedelta(days=1)).strftime("%Y-%m-%d")
            recent_1d = ticker_articles[ticker_articles["date"] >= cutoff_1d]

            # Last 7 days
            cutoff_7d = (now - timedelta(days=7)).strftime("%Y-%m-%d")
            recent_7d = ticker_articles[ticker_articles["date"] >= cutoff_7d]

            features.append({
                "ticker": ticker,
                "news_sentiment_1d": recent_1d["sentiment"].mean() if len(recent_1d) > 0 else 0.0,
                "news_sentiment_7d": recent_7d["sentiment"].mean() if len(recent_7d) > 0 else 0.0,
                "news_volume_1d": len(recent_1d),
                "news_volume_7d": len(recent_7d),
            })

        return pd.DataFrame(features)
