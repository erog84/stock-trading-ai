"""Retry decorator with exponential backoff."""

import time
import functools
from typing import Type

from src.utils.logger import logger


def retry(
    max_attempts: int = 3,
    backoff: float = 2.0,
    initial_delay: float = 1.0,
    exceptions: tuple[Type[Exception], ...] = (ConnectionError, TimeoutError, OSError),
):
    """Decorator that retries a function on failure with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts
        backoff: Multiplier for delay between retries
        initial_delay: Initial delay in seconds
        exceptions: Tuple of exception types to catch
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger.warning(
                            f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        delay *= backoff
                    else:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts: {e}")

            raise last_exception

        return wrapper
    return decorator
