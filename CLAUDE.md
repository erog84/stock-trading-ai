# Stock Market AI Trading Platform

## Project Overview
AI-powered stock trading platform that ingests daily market data from multiple sources, trains ML models to generate buy/sell signals, and will eventually execute trades via a brokerage API.

## Architecture
- **Data Layer**: `src/data/` - Multi-source data ingestion (Yahoo Finance, Alpha Vantage, Congressional trades, FRED, SEC EDGAR)
- **Model Layer**: `src/models/` - ML models starting with scikit-learn, evolving to deep learning and RL
- **Trading Layer**: `src/trading/` - Portfolio management, signal generation, broker API abstraction
- **API Layer**: `src/api/` - FastAPI backend (production)
- **Dashboard**: `src/dashboard/` - Streamlit prototype (phase 1), React frontend (phase 2)
- **Data Storage**: `data/` - raw, processed, and model artifacts

## Tech Stack
- **Language**: Python 3.11+
- **Data**: pandas, numpy, yfinance, alpha_vantage, fredapi
- **ML**: scikit-learn (phase 1), PyTorch (phase 2)
- **API**: FastAPI, uvicorn
- **Dashboard**: Streamlit (prototype), React + TradingView lightweight-charts (production)
- **Testing**: pytest, pytest-cov
- **Broker**: TBD (abstracted behind `src/trading/broker_api.py`)

## Conventions
- Use type hints on all function signatures
- Use dataclasses or pydantic models for structured data
- Config via environment variables loaded through `src/utils/config.py`
- Logging via `src/utils/logger.py` (structured JSON logging)
- All data fetchers implement a common interface pattern
- Tests mirror source structure in `tests/`
- Store API keys in `.env` (never commit)

## Custom Skills (Agents)
- `/plan` - Architecture planning and task breakdown
- `/develop` - Feature implementation
- `/test` - Test writing and execution
- `/document` - Documentation generation
- `/data-pipeline` - Data source management and feature engineering
- `/backtest` - Strategy backtesting and performance analysis

## Data Sources
1. Yahoo Finance (yfinance) - OHLCV, fundamentals
2. Alpha Vantage - Technical indicators, sector data
3. Congressional insider trading (QuiverQuant / Capitol Trades)
4. FRED - Economic indicators (interest rates, GDP, unemployment)
5. SEC EDGAR - Company filings

## ML Roadmap
1. Phase 1: Classical ML (Random Forest, XGBoost) for signal generation
2. Phase 2: LSTM/Transformer time series models
3. Phase 3: Reinforcement learning for portfolio optimization
