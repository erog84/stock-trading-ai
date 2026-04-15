"""Alerting via Discord webhook and email.

Both are optional and free. Discord webhooks require no API key,
just a webhook URL from your Discord server settings.
"""

from typing import Optional
from datetime import datetime
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

from src.utils.config import config
from src.utils.logger import logger


class AlertManager:
    """Sends alerts via Discord webhook and/or email."""

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        email_to: Optional[str] = None,
    ):
        self.webhook_url = webhook_url or config.alert_webhook_url
        self.email_to = email_to or config.alert_email_to

    def send(self, title: str, message: str, level: str = "info") -> None:
        """Send alert via all configured channels."""
        if self.webhook_url:
            self._send_discord(title, message, level)
        if self.email_to:
            self._send_email(title, message)

    def trade_executed(self, ticker: str, side: str, quantity: int, price: float, confidence: float) -> None:
        """Alert on trade execution."""
        emoji = "BUY" if side == "buy" else "SELL"
        self.send(
            title=f"Trade Executed: {emoji} {ticker}",
            message=(
                f"**{side.upper()}** {quantity} shares of {ticker}\n"
                f"Price: ${price:.2f}\n"
                f"Confidence: {confidence:.1%}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            ),
            level="info",
        )

    def daily_summary(self, portfolio_value: float, daily_pnl: float, trades_today: int) -> None:
        """Send daily P&L summary."""
        pnl_pct = daily_pnl / max(portfolio_value - daily_pnl, 1) * 100
        self.send(
            title="Daily Portfolio Summary",
            message=(
                f"Portfolio Value: ${portfolio_value:,.2f}\n"
                f"Daily P&L: ${daily_pnl:,.2f} ({pnl_pct:+.2f}%)\n"
                f"Trades Today: {trades_today}\n"
                f"Date: {datetime.now().strftime('%Y-%m-%d')}"
            ),
            level="info" if daily_pnl >= 0 else "warning",
        )

    def error_alert(self, component: str, error: str) -> None:
        """Alert on system error."""
        self.send(
            title=f"Error: {component}",
            message=f"Component: {component}\nError: {error}\nTime: {datetime.now().isoformat()}",
            level="error",
        )

    def _send_discord(self, title: str, message: str, level: str = "info") -> None:
        """Send Discord webhook notification."""
        colors = {"info": 3447003, "warning": 16776960, "error": 15158332}  # Blue, Yellow, Red

        payload = {
            "embeds": [{
                "title": title,
                "description": message,
                "color": colors.get(level, 3447003),
                "timestamp": datetime.utcnow().isoformat(),
                "footer": {"text": "Stock Trading AI"},
            }]
        }

        try:
            resp = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.warning(f"Discord alert failed: {e}")

    def _send_email(self, title: str, message: str) -> None:
        """Send email notification (using localhost SMTP or configured server)."""
        try:
            msg = MIMEMultipart()
            msg["Subject"] = f"[StockAI] {title}"
            msg["To"] = self.email_to
            msg["From"] = "stockai@localhost"
            msg.attach(MIMEText(message, "plain"))

            with smtplib.SMTP("localhost", 25, timeout=5) as server:
                server.send_message(msg)
        except Exception as e:
            logger.debug(f"Email alert failed (expected if no SMTP server): {e}")


# Module-level instance
alerts = AlertManager()
