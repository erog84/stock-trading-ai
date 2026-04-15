"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.portfolio import router as portfolio_router
from src.api.routes.data import router as data_router
from src.api.routes.models import router as models_router
from src.api.routes.signals import router as signals_router
from src.api.routes.broker import router as broker_router
from src.utils.config import config

app = FastAPI(
    title="Stock Trading AI Platform",
    description="AI-powered stock market analysis and trading signals",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(portfolio_router, prefix="/api/portfolio", tags=["Portfolio"])
app.include_router(data_router, prefix="/api/data", tags=["Data"])
app.include_router(models_router, prefix="/api/models", tags=["Models"])
app.include_router(signals_router, prefix="/api/signals", tags=["Signals"])
app.include_router(broker_router, prefix="/api/broker", tags=["Broker"])


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "version": "0.2.0",
        "broker_type": config.broker_type,
        "data_sources": {
            "yahoo_finance": True,
            "alpha_vantage": config.is_configured("alpha_vantage"),
            "fred": config.is_configured("fred"),
            "sec_edgar": True,
            "news_rss": True,
            "options": True,
            "congress": config.is_configured("quiver") or True,  # Free fallback available
            "alpaca": config.is_configured("alpaca"),
        },
    }


@app.get("/api/scheduler/status")
def scheduler_status():
    from src.utils.scheduler import scheduler
    return scheduler.get_status()


@app.post("/api/scheduler/run")
def run_pipeline(
    tickers: str = "AAPL,MSFT,GOOGL,AMZN,NVDA",
    model_type: str = "random_forest",
    execute: bool = False,
):
    """Manually trigger the daily pipeline."""
    from src.utils.scheduler import scheduler
    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    return scheduler.run_pipeline(ticker_list, model_type, execute_trades=execute)
