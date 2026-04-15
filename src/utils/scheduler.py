"""Daily pipeline scheduler.

Runs the full data-to-signals pipeline on a schedule.
Default: 30 minutes after market close (4:30 PM ET).
"""

from datetime import datetime
from typing import Optional, Callable
import threading

from src.utils.config import config
from src.utils.logger import logger


class PipelineScheduler:
    """Schedules and runs the daily trading pipeline."""

    def __init__(self):
        self._timer: Optional[threading.Timer] = None
        self._running = False
        self.last_run: Optional[datetime] = None
        self.last_status: str = "never_run"
        self.last_error: Optional[str] = None

    def run_pipeline(
        self,
        tickers: list[str],
        model_type: str = "random_forest",
        execute_trades: bool = False,
    ) -> dict:
        """Run the full daily pipeline.

        Steps:
        1. Fetch latest data from all sources
        2. Build feature matrix
        3. Generate predictions and signals
        4. Optionally execute trades
        5. Record results
        """
        from src.data.aggregator import DataAggregator
        from src.models.model_trainer import ModelTrainer
        from src.trading.signals import SignalGenerator
        from src.trading.executor import TradeExecutor

        self.last_status = "running"
        results = {"started_at": datetime.now().isoformat(), "steps": {}}

        try:
            # Step 1 & 2: Fetch data and build features
            logger.info("Pipeline: Building features...")
            aggregator = DataAggregator()
            features = aggregator.fetch_and_build_features(
                tickers, start_date="2024-01-01",
            )
            aggregator.save_features(features)
            results["steps"]["features"] = {
                "rows": len(features),
                "columns": len(features.columns),
            }

            # Step 3: Train/load model and generate signals
            logger.info("Pipeline: Training model...")
            trainer = ModelTrainer()
            model, metrics = trainer.train_model(features, model_type=model_type)
            results["steps"]["training"] = metrics

            logger.info("Pipeline: Generating signals...")
            latest = features.groupby("ticker").tail(1)
            current_prices = {row["ticker"]: row["close"] for _, row in latest.iterrows()}

            generator = SignalGenerator(model)
            signals = generator.generate_signals(
                features=latest,
                current_prices=current_prices,
                portfolio_value=100_000,
                current_positions=set(),
            )
            results["steps"]["signals"] = [
                {"ticker": s.ticker, "signal": s.signal.value, "confidence": s.confidence}
                for s in signals
            ]

            # Step 4: Execute trades (if enabled)
            if execute_trades and signals:
                logger.info("Pipeline: Executing trades...")
                executor = TradeExecutor()
                execution_results = executor.execute_signals(signals)
                results["steps"]["execution"] = execution_results
            else:
                results["steps"]["execution"] = "skipped" if not execute_trades else "no_signals"

            self.last_status = "success"
            self.last_run = datetime.now()
            results["completed_at"] = datetime.now().isoformat()
            results["status"] = "success"

            logger.info(f"Pipeline complete: {len(signals)} signals generated")

        except Exception as e:
            self.last_status = "error"
            self.last_error = str(e)
            results["status"] = "error"
            results["error"] = str(e)
            logger.error(f"Pipeline error: {e}")

        return results

    def get_status(self) -> dict:
        """Get scheduler status."""
        return {
            "enabled": config.scheduler_enabled,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "last_status": self.last_status,
            "last_error": self.last_error,
        }


# Module-level scheduler
scheduler = PipelineScheduler()
