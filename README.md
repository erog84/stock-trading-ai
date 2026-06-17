# Stock Trading AI Platform

An end-to-end, AI-powered stock-trading research platform in Python. It ingests
market data from 7+ sources, engineers features, trains classical and deep-learning
models to generate buy/sell signals, validates strategies with a walk-forward
backtester, and can execute paper or live trades through a pluggable broker
abstraction — all exposed via a FastAPI service and a Streamlit dashboard.

> ⚠️ **Disclaimer — educational use only.** This project is a personal research
> and learning exercise. It is **not financial advice**, makes no guarantee of
> profitability, and ships with `BROKER_TYPE=paper` (simulated) by default.
> **Always paper-trade first.** Trading real money carries real risk of loss;
> use the live broker mode entirely at your own risk.

## Features

- **Multi-source data ingestion** — Yahoo Finance, Alpha Vantage, FRED
  (macro indicators), SEC EDGAR filings, congressional trades, news/RSS
  sentiment, and options data, unified behind a common aggregator.
- **ML signal generation** — classical models (Random Forest, XGBoost) plus
  deep-learning time-series models (LSTM, Transformer) behind a shared trainer
  and ensemble layer.
- **Walk-forward backtesting** — strategy validation with portfolio simulation
  and performance metrics.
- **Pluggable broker layer** — a `broker_api` abstraction with a simulated
  paper broker and an Alpaca integration (paper or live).
- **FastAPI service + Streamlit dashboard** — programmatic API and an
  interactive UI, plus a no-terminal GUI launcher with health monitoring and
  auto-restart.
- **Automated daily pipeline** — fetch → train → signal → (optional) execute,
  schedulable via Windows Task Scheduler, with Discord/email alerts.
- **Tested** — `pytest` suite mirroring the source tree (data, models, trading,
  API, utils).

## Architecture

```
src/
├── data/        # Multi-source ingestion + aggregator
│   ├── yahoo_finance.py  alpha_vantage.py  fred_data.py  sec_edgar.py
│   ├── congress_trades.py  news_sentiment.py  options_data.py  sentiment.py
│   └── aggregator.py
├── models/      # ML models + training/ensembling
│   ├── random_forest.py  xgboost_model.py  lstm_model.py  transformer_model.py
│   ├── base_dl.py  model_trainer.py  ensemble.py  data_utils.py
├── trading/     # Signals, portfolio, backtesting, execution, brokers
│   ├── signals.py  portfolio.py  backtester.py  executor.py
│   ├── broker_api.py  alpaca_broker.py
├── api/         # FastAPI backend  (main.py)
├── dashboard/   # Streamlit UI     (app.py)
└── utils/       # config, logging, cache, retry, scheduler, alerts, validators
```

## Tech stack

- **Language:** Python 3.11+
- **Data:** pandas, numpy, pyarrow, yfinance, alpha-vantage, fredapi,
  sec-edgar-downloader, feedparser, beautifulsoup4
- **ML:** scikit-learn, XGBoost, (optional) PyTorch
- **Broker:** alpaca-py
- **API / UI:** FastAPI, uvicorn, websockets, pydantic, Streamlit, plotly
- **Utilities:** python-dotenv, loguru, pytz
- **Testing:** pytest, pytest-cov, pytest-asyncio, httpx

## Getting started

```bash
git clone https://github.com/erog84/stock-trading-ai.git
cd stock-trading-ai

python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
# Deep-learning models are optional:  pip install torch

cp .env.example .env   # then fill in keys (see below)
```

### API keys

Several sources are **free and need no key** (Yahoo Finance, SEC EDGAR, news
RSS, options via yfinance). For the rest, copy `.env.example` to `.env` and add
your keys — see [`docs/API_KEYS.md`](docs/API_KEYS.md) for where to get each.
Keys are loaded from `.env` (which is git-ignored) via `src/utils/config.py`;
**never commit real keys.** The broker defaults to simulated paper trading.

## Running

```bash
# Interactive dashboard (no terminal window, with auto-restart)
python launcher.pyw

# ...or run the dashboard directly
streamlit run src/dashboard/app.py

# FastAPI service
uvicorn src.api.main:app --reload

# One-shot daily pipeline (fetch data, train, generate signals)
python daily_pipeline.py
```

## Testing

```bash
pytest            # run the suite
pytest --cov=src  # with coverage
```

## ML roadmap

1. **Phase 1 — Classical ML:** Random Forest / XGBoost signal generation.
2. **Phase 2 — Deep learning:** LSTM / Transformer time-series models.
3. **Phase 3 — Reinforcement learning:** portfolio optimization.

## License

MIT — see [LICENSE](LICENSE).
