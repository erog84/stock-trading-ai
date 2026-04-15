"""Multi-source data aggregation and feature engineering."""

from typing import Optional
from pathlib import Path
import pandas as pd
import numpy as np

from src.data.yahoo_finance import YahooFinanceFetcher
from src.data.alpha_vantage import AlphaVantageFetcher
from src.data.congress_trades import CongressTradeFetcher
from src.data.fred_data import FredFetcher
from src.data.sec_edgar import SECEdgarFetcher
from src.data.news_sentiment import NewsSentimentFetcher
from src.data.options_data import OptionsDataFetcher
from src.utils.config import config
from src.utils.logger import logger


class DataAggregator:
    """Aggregates data from multiple sources and engineers features."""

    def __init__(self):
        self.yahoo = YahooFinanceFetcher()
        self.alpha_vantage = AlphaVantageFetcher()
        self.congress = CongressTradeFetcher()
        self.fred = FredFetcher()
        self.sec_edgar = SECEdgarFetcher()
        self.news = NewsSentimentFetcher()
        self.options = OptionsDataFetcher()
        self.data_dir = Path(config.data_dir)

    def fetch_and_build_features(
        self,
        tickers: list[str],
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch data from all sources and build feature matrix.

        Returns a DataFrame indexed by (date, ticker) with all features.
        """
        logger.info(f"Building feature matrix for {tickers} from {start_date}")

        # 1. Get base OHLCV data from Yahoo Finance
        ohlcv = self.yahoo.fetch(tickers, start_date, end_date)
        if ohlcv.empty:
            logger.error("No OHLCV data available")
            return pd.DataFrame()

        # 2. Add technical indicators (computed locally to avoid API limits)
        features = self._add_technical_indicators(ohlcv)

        # 3. Add congressional trading signals
        features = self._add_congress_signals(features, tickers, start_date, end_date)

        # 4. Add economic indicators
        features = self._add_economic_indicators(features, start_date, end_date)

        # 5. Add SEC EDGAR sentiment features
        features = self._add_sec_features(features, tickers, start_date, end_date)

        # 6. Add news sentiment features
        features = self._add_news_features(features, tickers, start_date, end_date)

        # 7. Add options-derived features
        features = self._add_options_features(features, tickers)

        # 8. Add temporal features
        features = self._add_temporal_features(features)

        # 9. Add target variable (next-day return)
        features = self._add_target(features)

        logger.info(f"Feature matrix built: {features.shape[0]} rows, {features.shape[1]} columns")
        return features

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators from OHLCV data."""
        result = df.copy()
        grouped = result.groupby("ticker")

        # Moving averages
        for window in [5, 10, 20, 50, 200]:
            result[f"sma_{window}"] = grouped["close"].transform(
                lambda x: x.rolling(window).mean()
            )
            result[f"ema_{window}"] = grouped["close"].transform(
                lambda x: x.ewm(span=window).mean()
            )

        # Price relative to moving averages
        for window in [20, 50, 200]:
            result[f"price_to_sma_{window}"] = result["close"] / result[f"sma_{window}"]

        # RSI (14-period)
        delta = grouped["close"].transform(lambda x: x.diff())
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        result["rsi_14"] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = grouped["close"].transform(lambda x: x.ewm(span=12).mean())
        ema_26 = grouped["close"].transform(lambda x: x.ewm(span=26).mean())
        result["macd"] = ema_12 - ema_26
        result["macd_signal"] = result.groupby("ticker")["macd"].transform(
            lambda x: x.ewm(span=9).mean()
        )
        result["macd_hist"] = result["macd"] - result["macd_signal"]

        # Bollinger Bands
        sma_20 = result["sma_20"]
        std_20 = grouped["close"].transform(lambda x: x.rolling(20).std())
        result["bb_upper"] = sma_20 + 2 * std_20
        result["bb_lower"] = sma_20 - 2 * std_20
        result["bb_width"] = (result["bb_upper"] - result["bb_lower"]) / sma_20
        result["bb_position"] = (result["close"] - result["bb_lower"]) / (
            result["bb_upper"] - result["bb_lower"]
        ).replace(0, np.nan)

        # Volume indicators
        result["volume_sma_20"] = grouped["volume"].transform(
            lambda x: x.rolling(20).mean()
        )
        result["volume_ratio"] = result["volume"] / result["volume_sma_20"].replace(0, np.nan)

        # Volatility
        result["daily_return"] = grouped["close"].transform(lambda x: x.pct_change())
        result["volatility_20"] = grouped["daily_return"].transform(
            lambda x: x.rolling(20).std() * np.sqrt(252)
        )

        # Average True Range
        high_low = result["high"] - result["low"]
        high_close = (result["high"] - result["close"].shift(1)).abs()
        low_close = (result["low"] - result["close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        result["atr_14"] = true_range.rolling(14).mean()

        # Momentum
        for period in [5, 10, 20]:
            result[f"momentum_{period}"] = grouped["close"].transform(
                lambda x: x.pct_change(period)
            )

        return result

    def _add_congress_signals(
        self,
        df: pd.DataFrame,
        tickers: list[str],
        start_date: str,
        end_date: Optional[str],
    ) -> pd.DataFrame:
        """Add congressional trading signal features."""
        try:
            congress_data = self.congress.fetch(tickers, start_date, end_date)
            if congress_data.empty:
                df["congress_buy_count"] = 0
                df["congress_sell_count"] = 0
                df["congress_net_signal"] = 0
                return df

            # Count buys and sells per ticker per day
            congress_data["is_buy"] = congress_data["transaction_type"].str.lower().str.contains("purchase", na=False)
            daily_signals = (
                congress_data.groupby(["disclosure_date", "ticker"])
                .agg(
                    congress_buy_count=("is_buy", "sum"),
                    congress_sell_count=("is_buy", lambda x: (~x).sum()),
                )
                .reset_index()
            )
            daily_signals["congress_net_signal"] = (
                daily_signals["congress_buy_count"] - daily_signals["congress_sell_count"]
            )

            # Merge with main dataframe
            df = df.reset_index()
            df = df.merge(
                daily_signals,
                left_on=["date", "ticker"],
                right_on=["disclosure_date", "ticker"],
                how="left",
            ).drop(columns=["disclosure_date"], errors="ignore")
            df = df.set_index("date")

            # Fill missing values
            for col in ["congress_buy_count", "congress_sell_count", "congress_net_signal"]:
                df[col] = df[col].fillna(0)

            # Rolling congressional activity (last 30 days)
            df["congress_activity_30d"] = df.groupby("ticker")["congress_net_signal"].transform(
                lambda x: x.rolling(30, min_periods=1).sum()
            )

        except Exception as e:
            logger.warning(f"Could not add congress signals: {e}")
            df["congress_buy_count"] = 0
            df["congress_sell_count"] = 0
            df["congress_net_signal"] = 0
            df["congress_activity_30d"] = 0

        return df

    def _add_economic_indicators(
        self,
        df: pd.DataFrame,
        start_date: str,
        end_date: Optional[str],
    ) -> pd.DataFrame:
        """Add macro economic indicators as features."""
        try:
            indicators = ["fed_funds_rate", "unemployment_rate", "treasury_10y",
                          "treasury_2y", "yield_spread", "vix"]
            econ = self.fred.fetch(indicators, start_date, end_date)

            if not econ.empty:
                # Forward fill economic data (released less frequently)
                econ = econ.ffill()

                # Merge with main dataframe
                df = df.join(econ, how="left")
                for col in econ.columns:
                    df[col] = df[col].ffill()

        except Exception as e:
            logger.warning(f"Could not add economic indicators: {e}")

        return df

    def _add_sec_features(
        self,
        df: pd.DataFrame,
        tickers: list[str],
        start_date: str,
        end_date: Optional[str],
    ) -> pd.DataFrame:
        """Add SEC EDGAR filing sentiment features."""
        try:
            sec_features = self.sec_edgar.get_sentiment_features(tickers, start_date, end_date)
            if not sec_features.empty and "ticker" in sec_features.columns:
                df = df.reset_index()
                df = df.merge(sec_features, on="ticker", how="left")
                df = df.set_index("date")
        except Exception as e:
            logger.warning(f"Could not add SEC features: {e}")

        # Ensure columns exist with defaults
        for col in ["sec_sentiment", "sec_filing_count_30d", "sec_uncertainty_pct"]:
            if col not in df.columns:
                df[col] = 0.0

        return df

    def _add_news_features(
        self,
        df: pd.DataFrame,
        tickers: list[str],
        start_date: str,
        end_date: Optional[str],
    ) -> pd.DataFrame:
        """Add news sentiment features."""
        try:
            news_features = self.news.get_sentiment_features(tickers, start_date, end_date)
            if not news_features.empty and "ticker" in news_features.columns:
                df = df.reset_index()
                df = df.merge(news_features, on="ticker", how="left")
                df = df.set_index("date")
        except Exception as e:
            logger.warning(f"Could not add news features: {e}")

        for col in ["news_sentiment_1d", "news_sentiment_7d", "news_volume_1d", "news_volume_7d"]:
            if col not in df.columns:
                df[col] = 0.0

        return df

    def _add_options_features(
        self,
        df: pd.DataFrame,
        tickers: list[str],
    ) -> pd.DataFrame:
        """Add options-derived features."""
        try:
            options_data = self.options.fetch(tickers, start_date="")
            if not options_data.empty and "ticker" in options_data.columns:
                options_cols = ["ticker", "put_call_ratio", "iv_avg", "iv_skew", "unusual_volume"]
                available = [c for c in options_cols if c in options_data.columns]
                df = df.reset_index()
                df = df.merge(options_data[available], on="ticker", how="left")
                df = df.set_index("date")
        except Exception as e:
            logger.warning(f"Could not add options features: {e}")

        for col in ["put_call_ratio", "iv_avg", "iv_skew", "unusual_volume"]:
            if col not in df.columns:
                df[col] = 0.0

        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        idx = df.index
        df["day_of_week"] = idx.dayofweek
        df["month"] = idx.month
        df["quarter"] = idx.quarter
        df["is_month_start"] = idx.is_month_start.astype(int)
        df["is_month_end"] = idx.is_month_end.astype(int)
        df["is_quarter_end"] = idx.is_quarter_end.astype(int)

        # Earnings season flags (roughly Jan/Apr/Jul/Oct)
        df["is_earnings_season"] = df["month"].isin([1, 4, 7, 10]).astype(int)

        return df

    def _add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add target variable: next-day return."""
        df["target_return_1d"] = df.groupby("ticker")["close"].transform(
            lambda x: x.pct_change().shift(-1)
        )
        # Binary target: 1 if positive return, 0 otherwise
        df["target_direction"] = (df["target_return_1d"] > 0).astype(int)
        return df

    def save_features(self, df: pd.DataFrame, filename: str = "features.parquet") -> Path:
        """Save feature matrix to disk."""
        path = self.data_dir / "processed" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        logger.info(f"Features saved to {path}")
        return path

    def load_features(self, filename: str = "features.parquet") -> pd.DataFrame:
        """Load feature matrix from disk."""
        path = self.data_dir / "processed" / filename
        if not path.exists():
            raise FileNotFoundError(f"Feature file not found: {path}")
        return pd.read_parquet(path)
