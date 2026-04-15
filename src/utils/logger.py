"""Structured logging configuration."""

import sys
from loguru import logger

from src.utils.config import config

# Remove default handler
logger.remove()

# Add structured handler
logger.add(
    sys.stderr,
    level=config.log_level,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)

# File handler for persistent logs
logger.add(
    "logs/app_{time:YYYY-MM-DD}.log",
    level="DEBUG",
    rotation="1 day",
    retention="30 days",
    serialize=True,
)
