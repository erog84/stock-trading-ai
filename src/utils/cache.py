"""File-based cache with TTL for API responses."""

import hashlib
import json
import time
from pathlib import Path
from typing import Optional, Any
import pandas as pd

from src.utils.config import config
from src.utils.logger import logger


class FileCache:
    """File-based cache with time-to-live expiration."""

    def __init__(self, cache_dir: Optional[str] = None, ttl_hours: Optional[int] = None):
        self.cache_dir = Path(cache_dir or config.data_dir) / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = (ttl_hours or config.cache_ttl_hours) * 3600

    def _make_key(self, source: str, params: dict) -> str:
        """Create a cache key from source and parameters."""
        raw = f"{source}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get_df(self, source: str, params: dict) -> Optional[pd.DataFrame]:
        """Retrieve a cached DataFrame if it exists and hasn't expired."""
        key = self._make_key(source, params)
        data_path = self.cache_dir / f"{key}.parquet"
        meta_path = self.cache_dir / f"{key}.meta"

        if not data_path.exists() or not meta_path.exists():
            return None

        # Check TTL
        try:
            meta = json.loads(meta_path.read_text())
            cached_at = meta.get("cached_at", 0)
            if time.time() - cached_at > self.ttl_seconds:
                logger.debug(f"Cache expired for {source}")
                return None
        except Exception:
            return None

        try:
            df = pd.read_parquet(data_path)
            logger.debug(f"Cache hit for {source} ({len(df)} rows)")
            return df
        except Exception:
            return None

    def set_df(self, source: str, params: dict, df: pd.DataFrame) -> None:
        """Cache a DataFrame."""
        key = self._make_key(source, params)
        data_path = self.cache_dir / f"{key}.parquet"
        meta_path = self.cache_dir / f"{key}.meta"

        try:
            df.to_parquet(data_path)
            meta_path.write_text(json.dumps({
                "cached_at": time.time(),
                "source": source,
                "params": params,
                "rows": len(df),
            }))
            logger.debug(f"Cached {len(df)} rows for {source}")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def get_json(self, source: str, params: dict) -> Optional[Any]:
        """Retrieve cached JSON data."""
        key = self._make_key(source, params)
        data_path = self.cache_dir / f"{key}.json"
        meta_path = self.cache_dir / f"{key}.meta"

        if not data_path.exists() or not meta_path.exists():
            return None

        try:
            meta = json.loads(meta_path.read_text())
            if time.time() - meta.get("cached_at", 0) > self.ttl_seconds:
                return None
            return json.loads(data_path.read_text())
        except Exception:
            return None

    def set_json(self, source: str, params: dict, data: Any) -> None:
        """Cache JSON data."""
        key = self._make_key(source, params)
        data_path = self.cache_dir / f"{key}.json"
        meta_path = self.cache_dir / f"{key}.meta"

        try:
            data_path.write_text(json.dumps(data))
            meta_path.write_text(json.dumps({
                "cached_at": time.time(),
                "source": source,
                "params": params,
            }))
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def clear(self, source: Optional[str] = None) -> int:
        """Clear cache. If source given, only clear that source's entries."""
        count = 0
        for meta_path in self.cache_dir.glob("*.meta"):
            try:
                if source:
                    meta = json.loads(meta_path.read_text())
                    if meta.get("source") != source:
                        continue

                key = meta_path.stem
                meta_path.unlink(missing_ok=True)
                (self.cache_dir / f"{key}.parquet").unlink(missing_ok=True)
                (self.cache_dir / f"{key}.json").unlink(missing_ok=True)
                count += 1
            except Exception:
                pass

        logger.info(f"Cleared {count} cache entries" + (f" for {source}" if source else ""))
        return count


# Module-level cache instance
cache = FileCache()
