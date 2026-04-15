"""Tests for file-based cache."""

import pytest
import tempfile
import pandas as pd

from src.utils.cache import FileCache


class TestFileCache:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache = FileCache(cache_dir=self.tmpdir, ttl_hours=24)

    def test_cache_miss(self):
        result = self.cache.get_df("test", {"key": "value"})
        assert result is None

    def test_cache_hit(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        self.cache.set_df("test", {"key": "value"}, df)
        result = self.cache.get_df("test", {"key": "value"})
        assert result is not None
        assert len(result) == 3

    def test_cache_json(self):
        data = {"tickers": ["AAPL", "MSFT"], "count": 42}
        self.cache.set_json("test", {"key": "value"}, data)
        result = self.cache.get_json("test", {"key": "value"})
        assert result == data

    def test_different_params_different_cache(self):
        df1 = pd.DataFrame({"a": [1]})
        df2 = pd.DataFrame({"a": [2, 3]})
        self.cache.set_df("test", {"key": "a"}, df1)
        self.cache.set_df("test", {"key": "b"}, df2)
        assert len(self.cache.get_df("test", {"key": "a"})) == 1
        assert len(self.cache.get_df("test", {"key": "b"})) == 2

    def test_clear_all(self):
        self.cache.set_df("test", {"key": "1"}, pd.DataFrame({"a": [1]}))
        self.cache.set_df("test", {"key": "2"}, pd.DataFrame({"a": [2]}))
        count = self.cache.clear()
        assert count == 2
        assert self.cache.get_df("test", {"key": "1"}) is None

    def test_expired_cache(self):
        import time
        cache = FileCache(cache_dir=self.tmpdir, ttl_hours=0)  # 0 TTL = always expired
        cache.ttl_seconds = -1  # Force everything to be expired
        df = pd.DataFrame({"a": [1]})
        cache.set_df("test", {"key": "value"}, df)
        result = cache.get_df("test", {"key": "value"})
        assert result is None  # Should be expired


class TestValidators:
    def test_validate_ohlcv_valid(self):
        from src.utils.validators import validate_ohlcv
        df = pd.DataFrame({
            "open": [150.0], "high": [155.0], "low": [149.0],
            "close": [153.0], "volume": [1000000],
        })
        issues = validate_ohlcv(df)
        assert len(issues) == 0

    def test_validate_ohlcv_high_lt_low(self):
        from src.utils.validators import validate_ohlcv
        df = pd.DataFrame({
            "open": [150.0], "high": [148.0], "low": [149.0],
            "close": [153.0], "volume": [1000000],
        })
        issues = validate_ohlcv(df)
        assert any("high < low" in i for i in issues)

    def test_validate_signal(self):
        from src.utils.validators import validate_signal
        from src.trading import Trade, Signal
        from datetime import datetime
        trade = Trade("AAPL", Signal.BUY, 150.0, 10, datetime.now(), 0.75, "test")
        issues = validate_signal(trade)
        assert len(issues) == 0

    def test_validate_signal_invalid(self):
        from src.utils.validators import validate_signal
        from src.trading import Trade, Signal
        from datetime import datetime
        trade = Trade("", Signal.BUY, -10.0, 0, datetime.now(), 2.0, "test")
        issues = validate_signal(trade)
        assert len(issues) > 0
