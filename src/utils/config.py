"""Configuration management via environment variables."""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    # Data sources
    alpha_vantage_api_key: str = field(default_factory=lambda: os.getenv("ALPHA_VANTAGE_API_KEY", ""))
    fred_api_key: str = field(default_factory=lambda: os.getenv("FRED_API_KEY", ""))
    quiver_api_key: str = field(default_factory=lambda: os.getenv("QUIVER_API_KEY", ""))

    # Alpaca broker
    alpaca_api_key: str = field(default_factory=lambda: os.getenv("ALPACA_API_KEY", ""))
    alpaca_api_secret: str = field(default_factory=lambda: os.getenv("ALPACA_API_SECRET", ""))
    alpaca_base_url: str = field(default_factory=lambda: os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"))

    # Legacy broker fields (for generic broker interface)
    broker_api_key: str = field(default_factory=lambda: os.getenv("BROKER_API_KEY", ""))
    broker_api_secret: str = field(default_factory=lambda: os.getenv("BROKER_API_SECRET", ""))
    broker_type: str = field(default_factory=lambda: os.getenv("BROKER_TYPE", "paper"))  # paper, alpaca_paper, alpaca_live

    # Alerts
    alert_webhook_url: str = field(default_factory=lambda: os.getenv("ALERT_WEBHOOK_URL", ""))
    alert_email_to: str = field(default_factory=lambda: os.getenv("ALERT_EMAIL_TO", ""))

    # Scheduler
    scheduler_enabled: bool = field(default_factory=lambda: os.getenv("SCHEDULER_ENABLED", "false").lower() == "true")

    # Cache
    cache_ttl_hours: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL_HOURS", "24")))

    # General
    env: str = field(default_factory=lambda: os.getenv("ENV", "development"))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    data_dir: str = field(default_factory=lambda: os.getenv("DATA_DIR", "data"))

    def is_configured(self, source: str) -> bool:
        """Check if a data source or service has its required keys configured."""
        checks = {
            "alpha_vantage": bool(self.alpha_vantage_api_key),
            "fred": bool(self.fred_api_key),
            "quiver": bool(self.quiver_api_key),
            "alpaca": bool(self.alpaca_api_key and self.alpaca_api_secret),
            "yahoo": True,
            "sec_edgar": True,
            "news_rss": True,
            "options": True,
        }
        return checks.get(source, False)


config = Config()
