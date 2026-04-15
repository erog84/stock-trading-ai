# API Keys Setup Guide

## Sources That Work Without Any API Key

These are free and require no setup:

| Source | What It Provides | Notes |
|--------|-----------------|-------|
| **Yahoo Finance** | Daily OHLCV, fundamentals, options chains | Via `yfinance` library |
| **SEC EDGAR** | Company filings (10-K, 10-Q, 8-K) + sentiment | Via `sec-edgar-downloader` |
| **News RSS** | Financial news headlines + sentiment | Google News, MarketWatch, Yahoo RSS |
| **Options Data** | Put/call ratios, IV, unusual activity | Via `yfinance` options chains |
| **Congress Trades** | House member stock transactions | Free via House Stock Watcher data |

## Sources That Need a Free API Key

### Alpha Vantage (25 requests/day free)

Provides: Technical indicators (RSI, MACD), sector performance, earnings data.

1. Go to https://www.alphavantage.co/support/#api-key
2. Enter your email and click "GET FREE API KEY"
3. Copy the key and add to `.env`:
   ```
   ALPHA_VANTAGE_API_KEY=your_key_here
   ```

**Note**: Free tier allows 25 API calls per day. The app caches responses to maximize this limit.

### FRED (Federal Reserve Economic Data)

Provides: Interest rates, unemployment, GDP, CPI, VIX, treasury yields, consumer sentiment.

1. Go to https://fred.stlouisfed.org/docs/api/api_key.html
2. Click "Request an API Key" (requires free FRED account)
3. Copy the key and add to `.env`:
   ```
   FRED_API_KEY=your_key_here
   ```

### Alpaca (Free Paper Trading)

Provides: Paper trading execution, live quotes, order management. No account minimum.

1. Go to https://app.alpaca.markets/signup
2. Create a free account (email verification required)
3. Once logged in, go to **Paper Trading** in the left sidebar
4. Click **API Keys** > **Generate New Key**
5. Copy both the API Key and Secret Key
6. Add to `.env`:
   ```
   ALPACA_API_KEY=your_api_key
   ALPACA_API_SECRET=your_secret_key
   ALPACA_BASE_URL=https://paper-api.alpaca.markets
   BROKER_TYPE=alpaca_paper
   ```

**Important**: Start with paper trading (`BROKER_TYPE=alpaca_paper`) before switching to live trading.
For live trading, change:
   ```
   ALPACA_BASE_URL=https://api.alpaca.markets
   BROKER_TYPE=alpaca_live
   ```

## Optional Paid Sources

### QuiverQuant (Congressional Trading - Enhanced)

The free House Stock Watcher fallback covers House trades. QuiverQuant adds Senate trades and historical data.

1. Go to https://www.quiverquant.com/
2. Subscribe to a plan
3. Add to `.env`:
   ```
   QUIVER_API_KEY=your_key_here
   ```

## Optional: Alerts

### Discord Webhook (Free)

Get trade and error notifications in Discord:

1. Open Discord, go to your server
2. Right-click a channel > Edit Channel > Integrations > Webhooks
3. Create a new webhook, copy the URL
4. Add to `.env`:
   ```
   ALERT_WEBHOOK_URL=https://discord.com/api/webhooks/...
   ```

### Email Alerts

For email alerts via local SMTP:
```
ALERT_EMAIL_TO=your@email.com
```

## Quick Start

Minimum viable setup (no keys needed):
```bash
cp .env.example .env
# That's it! Yahoo Finance, SEC EDGAR, News RSS, Options, and Congress trades all work without keys.
```

Recommended setup (add free keys):
```bash
cp .env.example .env
# Edit .env and add:
# - ALPHA_VANTAGE_API_KEY (for technical indicators)
# - FRED_API_KEY (for economic data)
# - ALPACA keys (for paper trading)
```
