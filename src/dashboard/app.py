"""Streamlit dashboard for the Stock Trading AI Platform."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.yahoo_finance import YahooFinanceFetcher
from src.data.aggregator import DataAggregator
from src.models.model_trainer import ModelTrainer
from src.trading.portfolio import Portfolio
from src.trading.backtester import Backtester, BacktestConfig
from src.utils.config import config

st.set_page_config(page_title="Stock Trading AI", page_icon="📈", layout="wide")


@st.dialog("Stock Trading AI Platform - How It Works", width="large")
def show_guide():
    st.markdown("""
### Getting Started

Work through the tabs **left to right** - each step builds on the previous one.

---

**1. Market Data** - *Fetch raw stock prices*
> Click **Fetch Data** to pull daily price data (open, high, low, close, volume) for your tickers from Yahoo Finance. This shows you candlestick charts so you can visually inspect each stock. No API key needed.

**2. Model Training** - *Build features and train the AI*
> Click **Build Features & Train Model** to pull data from all sources (Yahoo, SEC filings, news, options), compute 50+ technical indicators, and train your selected ML model. The model learns patterns that predict whether a stock will go up or down tomorrow. Check the feature importance chart to see what the model considers most predictive.

**3. Backtesting** - *Test strategy on historical data*
> Simulates what would have happened if you followed the model's signals over the lookback period. Adjust position size and max positions to test different risk levels. Key metrics: **Sharpe Ratio** (>1 is good), **Max Drawdown** (worst-case loss), and **Win Rate** (% of profitable trades).

**4. Signals** - *Get today's buy/sell recommendations*
> Uses the trained model to generate live trading signals for your tickers right now. Only shows signals above your confidence threshold. Each signal includes the model's reasoning (technical indicators, congressional activity, etc.).

**5. Congress Trades** - *Track what politicians are buying*
> Scans congressional financial disclosures for stocks being purchased by multiple members of Congress. When several politicians buy the same stock around the same time, it can indicate insider knowledge.

**6. Options** - *Scan for unusual options activity*
> Checks options chains for abnormal volume, put/call ratios, and implied volatility skew. Unusual options activity often precedes big price moves.

**7. Broker** - *Connect to Alpaca for live trading*
> Shows your Alpaca paper trading account (positions, orders, balance). Requires free Alpaca API keys in your `.env` file. Start with paper trading before risking real money.

**8. Pipeline** - *Run everything in one click*
> Runs the full pipeline: fetch data, build features, train model, generate signals, and optionally execute trades. This is what the daily automated scheduler runs at 4:35 PM.

---

### Quick Start
1. Go to **Market Data** and click **Fetch Data**
2. Go to **Model Training** and click **Build Features & Train Model**
3. Go to **Backtesting** and click **Run Backtest**
4. Go to **Signals** to see today's recommendations
    """)
    if st.button("Got it!", use_container_width=True):
        st.rerun()


col_title, col_help = st.columns([6, 1])
with col_title:
    st.title("Stock Trading AI Platform")
with col_help:
    st.write("")  # spacer
    if st.button("How to Use", type="secondary"):
        show_guide()

# Sidebar
st.sidebar.header("Configuration")

tickers_input = st.sidebar.text_input(
    "Tickers (comma-separated)", value="AAPL, MSFT, GOOGL, AMZN, NVDA",
    help="Enter stock ticker symbols separated by commas. These are used across all tabs for data fetching, model training, and signal generation.",
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

lookback_days = st.sidebar.slider(
    "Lookback (days)", 30, 1000, 365,
    help="How many days of historical data to fetch. More data gives the model more to learn from, but takes longer to process. 365 days (1 year) is a good default.",
)
start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

_model_options = ["random_forest", "xgboost", "ensemble"]
try:
    from src.models.base_dl import HAS_TORCH
    if HAS_TORCH:
        _model_options.extend(["lstm", "transformer"])
except Exception:
    pass
model_type = st.sidebar.selectbox(
    "Model", _model_options,
    help="**random_forest** - Fast, interpretable, good baseline. **xgboost** - Usually more accurate, handles complex patterns. **ensemble** - Combines multiple models by averaging their predictions. **lstm/transformer** - Deep learning models for sequential patterns (requires PyTorch).",
)
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.5, 0.9, 0.6, 0.05,
    help="Minimum model confidence to generate a buy/sell signal. Higher = fewer but stronger signals, Lower = more signals but more noise. 0.6 means the model must be at least 60% sure before recommending a trade.",
)

# Data source status
st.sidebar.markdown("---")
st.sidebar.subheader("Data Sources")
sources = {
    "Yahoo Finance": ("yes", True),
    "SEC EDGAR": ("yes", True),
    "News RSS": ("yes", True),
    "Options": ("yes", True),
    "Congress Trades": ("free fallback", True),
    "Alpha Vantage": ("key needed", config.is_configured("alpha_vantage")),
    "FRED": ("key needed", config.is_configured("fred")),
    "Alpaca Broker": ("key needed", config.is_configured("alpaca")),
}
for name, (status, configured) in sources.items():
    icon = "✅" if configured else "⚙️"
    st.sidebar.text(f"{icon} {name} ({status})")

# Tabs
tab_market, tab_model, tab_backtest, tab_signals, tab_congress, tab_options, tab_broker, tab_pipeline = st.tabs([
    "Market Data", "Model Training", "Backtesting", "Signals",
    "Congress Trades", "Options", "Broker", "Pipeline",
])

# --- Market Data Tab ---
with tab_market:
    st.header("Market Data")
    st.caption("Fetch daily OHLCV price data from Yahoo Finance. View candlestick charts for each ticker. This is your starting point - fetch data here before training models.")

    if st.button("Fetch Data", key="fetch"):
        with st.spinner("Fetching market data..."):
            yahoo = YahooFinanceFetcher()
            data = yahoo.fetch(tickers, start_date)

            if not data.empty:
                st.success(f"Fetched {len(data)} rows for {len(tickers)} tickers")

                for ticker in tickers:
                    ticker_data = data[data["ticker"] == ticker]
                    if ticker_data.empty:
                        continue

                    st.subheader(ticker)
                    fig = go.Figure(data=[go.Candlestick(
                        x=ticker_data.index,
                        open=ticker_data["open"], high=ticker_data["high"],
                        low=ticker_data["low"], close=ticker_data["close"],
                        name=ticker,
                    )])
                    fig.update_layout(title=f"{ticker} Price", yaxis_title="Price ($)",
                                      xaxis_rangeslider_visible=False, height=400)
                    st.plotly_chart(fig, use_container_width=True)

                st.session_state["market_data"] = data
            else:
                st.error("No data fetched")

# --- Model Training Tab ---
with tab_model:
    st.header("Model Training")
    st.caption("Pulls data from all sources, computes 50+ features (technical indicators, sentiment, options), and trains your AI model. Shows feature importance so you can see what drives predictions.")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Build Features & Train Model"):
            with st.spinner(f"Training {model_type} model..."):
                try:
                    aggregator = DataAggregator()
                    features = aggregator.fetch_and_build_features(tickers, start_date)

                    if features.empty:
                        st.error("No features built")
                    else:
                        aggregator.save_features(features)
                        st.success(f"Features: {features.shape[0]} rows, {features.shape[1]} columns")
                        st.session_state["features"] = features

                        trainer = ModelTrainer()
                        model, metrics = trainer.train_model(features, model_type=model_type)
                        st.session_state["model"] = model
                        st.session_state["metrics"] = metrics
                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    with col2:
        if "metrics" in st.session_state:
            st.subheader("Model Metrics")
            metrics = st.session_state["metrics"]
            for key, value in metrics.items():
                if isinstance(value, float):
                    st.metric(key, f"{value:.4f}")
                elif isinstance(value, (int, str)):
                    st.metric(key, value)

    # Training history for DL models
    if "model" in st.session_state and hasattr(st.session_state["model"], "training_history"):
        history = st.session_state["model"].training_history
        if history:
            st.subheader("Training History")
            hist_df = pd.DataFrame(history)
            fig = px.line(hist_df, x="epoch", y=["train_loss", "val_loss"], title="Loss Curves")
            st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    if "model" in st.session_state:
        st.subheader("Feature Importance (Top 20)")
        importance = st.session_state["model"].get_feature_importance()
        top_20 = dict(list(importance.items())[:20])
        if top_20:
            fig = px.bar(x=list(top_20.values()), y=list(top_20.keys()), orientation="h",
                         title="Feature Importance")
            fig.update_layout(height=500, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

# --- Backtesting Tab ---
with tab_backtest:
    st.header("Backtesting")
    st.caption("Simulate trading over historical data to see how the model would have performed. Requires training a model first. Adjusts for slippage and commissions.")

    col1, col2, col3 = st.columns(3)
    with col1:
        bt_initial_cash = st.number_input(
            "Initial Cash ($)", value=100_000, step=10_000,
            help="Starting portfolio value for the backtest simulation.",
        )
    with col2:
        bt_position_size = st.slider(
            "Position Size (%)", 1, 20, 5,
            help="Maximum percentage of portfolio value to put into a single stock. Lower = less risk per trade, Higher = more concentrated bets. 5% is conservative.",
        )
    with col3:
        bt_max_positions = st.slider(
            "Max Positions", 1, 50, 20,
            help="Maximum number of stocks to hold at once. More positions = more diversification but thinner per-stock allocation.",
        )

    if st.button("Run Backtest"):
        if "features" not in st.session_state:
            st.warning("Build features first in the Model Training tab")
        else:
            with st.spinner("Running backtest..."):
                try:
                    from src.models.random_forest import RandomForestModel
                    from src.models.xgboost_model import XGBoostModel

                    model_map = {"random_forest": RandomForestModel, "xgboost": XGBoostModel}

                    try:
                        from src.models.lstm_model import LSTMModel
                        from src.models.transformer_model import TransformerModel
                        model_map["lstm"] = LSTMModel
                        model_map["transformer"] = TransformerModel
                    except (ImportError, NameError):
                        pass

                    model_cls = model_map.get(model_type, RandomForestModel)
                    model = model_cls()

                    bt_config = BacktestConfig(
                        initial_cash=bt_initial_cash,
                        confidence_threshold=confidence_threshold,
                        position_size_pct=bt_position_size / 100,
                        max_positions=bt_max_positions,
                    )
                    backtester = Backtester(bt_config)
                    results = backtester.run(st.session_state["features"], model)

                    st.session_state["backtest_results"] = results

                    # Summary
                    st.subheader("Performance Summary")
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("Total Return", f"{results.get('total_return_pct', 0):.2f}%",
                                  help="Total percentage gain or loss over the backtest period.")
                    with cols[1]:
                        st.metric("Sharpe Ratio", f"{results.get('sharpe_ratio', 0):.2f}",
                                  help="Risk-adjusted return. Above 1.0 is good, above 2.0 is very good. Measures return per unit of risk taken.")
                    with cols[2]:
                        st.metric("Max Drawdown", f"{results.get('max_drawdown_pct', 0):.2f}%",
                                  help="Largest peak-to-trough decline during the backtest. Shows worst-case loss you would have experienced.")
                    with cols[3]:
                        st.metric("Win Rate", f"{results.get('win_rate', 0):.1f}%",
                                  help="Percentage of closed trades that were profitable. Above 50% means more winners than losers.")

                    # Equity curve
                    daily = results.get("daily_values")
                    if daily is not None and not daily.empty:
                        fig = px.line(daily, x="date", y="total_value", title="Portfolio Equity Curve")
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)

                    # Trade log
                    trades_df = results.get("trades")
                    if trades_df is not None and not trades_df.empty:
                        st.subheader(f"Trades ({len(trades_df)})")
                        st.dataframe(trades_df, use_container_width=True)

                except Exception as e:
                    st.error(f"Backtest error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

# --- Signals Tab ---
with tab_signals:
    st.header("Trading Signals")
    st.caption("Generate live buy/sell recommendations using the trained model on the latest data. Requires a trained model. Signals include confidence scores and reasoning.")

    if st.button("Generate Signals"):
        model_path = Path("data/models") / f"{model_type}_latest.joblib"
        if not model_path.exists():
            st.warning("Train a model first in the Model Training tab")
        else:
            with st.spinner("Generating signals..."):
                try:
                    from src.models.random_forest import RandomForestModel
                    from src.models.xgboost_model import XGBoostModel
                    from src.trading.signals import SignalGenerator

                    model_map = {"random_forest": RandomForestModel, "xgboost": XGBoostModel}
                    try:
                        from src.models.lstm_model import LSTMModel
                        from src.models.transformer_model import TransformerModel
                        model_map["lstm"] = LSTMModel
                        model_map["transformer"] = TransformerModel
                    except (ImportError, NameError):
                        pass

                    model_cls = model_map.get(model_type, RandomForestModel)
                    model = model_cls()
                    model.load(str(model_path))

                    aggregator = DataAggregator()
                    features = aggregator.fetch_and_build_features(tickers, start_date)
                    latest = features.groupby("ticker").tail(1)
                    current_prices = {row["ticker"]: row["close"] for _, row in latest.iterrows()}

                    generator = SignalGenerator(model, confidence_threshold=confidence_threshold)
                    signals = generator.generate_signals(
                        features=latest, current_prices=current_prices,
                        portfolio_value=100_000, current_positions=set(),
                    )

                    if signals:
                        for s in signals:
                            color = "green" if s.signal.value == "buy" else "red"
                            st.markdown(
                                f"**:{color}[{s.signal.value.upper()}]** {s.ticker} "
                                f"@ ${s.price:.2f} | Confidence: {s.confidence:.1%} | {s.reason}"
                            )
                    else:
                        st.info("No signals above confidence threshold")
                except Exception as e:
                    st.error(f"Error: {e}")

# --- Congress Trades Tab ---
with tab_congress:
    st.header("Congressional Trading Activity")
    st.caption("Scans public financial disclosures from members of Congress. Flags stocks being bought by multiple politicians - potential insider knowledge signals.")
    col1, col2 = st.columns(2)
    with col1:
        congress_days = st.slider(
            "Lookback (days)", 7, 180, 30, key="congress_days",
            help="How far back to look for congressional stock trades. Congress members must disclose trades within 45 days.",
        )
    with col2:
        min_trades = st.slider(
            "Min. congress members trading", 1, 10, 3,
            help="Minimum number of different congress members buying the same stock to generate a signal. Higher = stronger consensus signal.",
        )

    if st.button("Check Congress Signals"):
        with st.spinner("Checking congressional trades..."):
            from src.data.congress_trades import CongressTradeFetcher
            congress = CongressTradeFetcher()
            signals = congress.get_buy_signals(min_trades=min_trades, days=congress_days)
            if not signals.empty:
                st.dataframe(signals, use_container_width=True)
            else:
                st.info("No congressional trading signals found")

# --- Options Tab ---
with tab_options:
    st.header("Options Activity")
    st.caption("Analyzes options chains for put/call ratios, implied volatility, and unusual volume. High unusual volume (>2x) often signals that informed traders expect a big move.")

    if st.button("Scan Options"):
        with st.spinner("Scanning options chains..."):
            from src.data.options_data import OptionsDataFetcher
            options = OptionsDataFetcher()
            data = options.fetch(tickers, start_date="")

            if not data.empty:
                st.dataframe(data, use_container_width=True)

                # Highlight unusual activity
                unusual = data[data["unusual_volume"] >= 2.0]
                if not unusual.empty:
                    st.subheader("Unusual Options Activity")
                    for _, row in unusual.iterrows():
                        st.warning(
                            f"**{row['ticker']}**: Vol/OI ratio {row['unusual_volume']:.1f}x | "
                            f"P/C ratio: {row['put_call_ratio']:.2f} | IV: {row['iv_avg']:.1%}"
                        )
            else:
                st.info("No options data available")

# --- Broker Tab ---
with tab_broker:
    st.header("Broker Connection")
    st.caption("Connect to Alpaca for paper or live trading. View account balance, positions, and recent orders. Requires free Alpaca API keys configured in .env.")

    if config.is_configured("alpaca"):
        st.success(f"Alpaca configured ({config.broker_type})")

        if st.button("Check Broker Status"):
            try:
                from src.trading.alpaca_broker import AlpacaBroker
                broker = AlpacaBroker()
                account = broker.get_account_info()
                cols = st.columns(4)
                with cols[0]:
                    st.metric("Cash", f"${account['cash']:,.2f}")
                with cols[1]:
                    st.metric("Portfolio Value", f"${account['total_value']:,.2f}")
                with cols[2]:
                    st.metric("Buying Power", f"${account.get('buying_power', 0):,.2f}")
                with cols[3]:
                    st.metric("Positions", account.get("n_positions", 0))

                # Show positions
                positions = broker.get_positions()
                if positions:
                    st.subheader("Positions")
                    st.dataframe(pd.DataFrame(positions), use_container_width=True)

                # Show recent orders
                orders = broker.get_recent_orders(limit=10)
                if orders:
                    st.subheader("Recent Orders")
                    st.dataframe(pd.DataFrame(orders), use_container_width=True)
            except Exception as e:
                st.error(f"Broker error: {e}")
    else:
        st.info(
            "Alpaca broker not configured.\n\n"
            "1. Sign up at https://app.alpaca.markets/signup\n"
            "2. Generate API keys (Paper Trading > API Keys)\n"
            "3. Add to `.env`:\n"
            "```\n"
            "ALPACA_API_KEY=your_key\n"
            "ALPACA_API_SECRET=your_secret\n"
            "ALPACA_BASE_URL=https://paper-api.alpaca.markets\n"
            "BROKER_TYPE=alpaca_paper\n"
            "```"
        )

# --- Pipeline Tab ---
with tab_pipeline:
    st.header("Daily Pipeline")
    st.caption("Run the full automated pipeline in one click: fetch data from all sources, build features, train model, generate signals, and optionally execute trades. This is what runs daily at 4:35 PM via Windows Task Scheduler.")

    pipeline_tickers = st.text_input(
        "Pipeline tickers", value=tickers_input, key="pipeline_tickers",
        help="Tickers to include in the daily pipeline run. This fetches data, builds features, trains the model, and generates signals for all listed tickers.",
    )
    pipeline_model = st.selectbox(
        "Pipeline model", _model_options, key="pipeline_model",
        help="Which model to use for the pipeline run. The model is retrained on the latest data each time.",
    )
    execute_trades = st.checkbox(
        "Execute trades (requires broker)", value=False,
        help="If checked, generated signals will be automatically submitted to your broker (Alpaca). Leave unchecked to only generate signals without trading. Requires Alpaca API keys in .env.",
    )

    if st.button("Run Pipeline Now"):
        with st.spinner("Running full pipeline..."):
            from src.utils.scheduler import scheduler
            ticker_list = [t.strip().upper() for t in pipeline_tickers.split(",") if t.strip()]
            results = scheduler.run_pipeline(ticker_list, pipeline_model, execute_trades=execute_trades)

            if results.get("status") == "success":
                st.success("Pipeline completed successfully!")

                # Show signals
                signals = results.get("steps", {}).get("signals", [])
                if signals:
                    st.subheader("Generated Signals")
                    st.dataframe(pd.DataFrame(signals), use_container_width=True)

                # Show training metrics
                metrics = results.get("steps", {}).get("training", {})
                if metrics:
                    st.subheader("Model Metrics")
                    st.json(metrics)
            else:
                st.error(f"Pipeline failed: {results.get('error', 'Unknown error')}")

    # Scheduler status
    st.subheader("Scheduler Status")
    from src.utils.scheduler import scheduler
    status = scheduler.get_status()
    st.json(status)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Stock Trading AI Platform** v0.2.0")
