"""Daily automated pipeline - run via Windows Task Scheduler.

Fetches latest market data, trains models, generates signals.
Results are logged to data/pipeline_results.json.
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Ensure project is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.scheduler import PipelineScheduler
from src.utils.logger import logger

# Default tickers to track
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
           "JPM", "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD"]

RESULTS_FILE = Path("data/pipeline_results.json")


def main():
    logger.info("=" * 60)
    logger.info(f"Daily pipeline started at {datetime.now().isoformat()}")
    logger.info(f"Tickers: {TICKERS}")
    logger.info("=" * 60)

    scheduler = PipelineScheduler()

    results = scheduler.run_pipeline(
        tickers=TICKERS,
        model_type="random_forest",
        execute_trades=False,  # Set to True once you trust the model + have Alpaca configured
    )

    # Save results
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Load existing history
    history = []
    if RESULTS_FILE.exists():
        try:
            history = json.loads(RESULTS_FILE.read_text())
        except Exception:
            history = []

    # Serialize results (strip DataFrames)
    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = v
        elif isinstance(v, (str, int, float, bool)):
            serializable[k] = v
        else:
            serializable[k] = str(v)

    history.append(serializable)

    # Keep last 90 days of results
    history = history[-90:]
    RESULTS_FILE.write_text(json.dumps(history, indent=2, default=str))

    # Send alert if configured
    try:
        from src.utils.alerts import alerts
        signals = results.get("steps", {}).get("signals", [])
        if signals:
            signal_text = "\n".join(
                f"  {s['signal'].upper()} {s['ticker']} (confidence: {s['confidence']:.1%})"
                for s in signals
            )
            alerts.send(
                title=f"Daily Signals - {datetime.now().strftime('%Y-%m-%d')}",
                message=f"Generated {len(signals)} signals:\n{signal_text}",
            )
    except Exception as e:
        logger.warning(f"Alert failed: {e}")

    status = results.get("status", "unknown")
    logger.info(f"Pipeline finished with status: {status}")
    return 0 if status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
