# BTC Futures Lab (fresh deployment)

Read-only Streamlit dashboard using public MEXC BTC_USDT futures candles. No exchange account, API key, or order placement.

## Files
- `app.py` — dashboard and model
- `requirements.txt` — Python dependencies

## Deploy from iPhone
1. Download and extract the ZIP.
2. In GitHub, create a new repository named `btc-futures-lab-clean`.
3. Upload `app.py` and `requirements.txt` directly to the repository's main page (the root). Do not upload the ZIP itself.
4. Open https://share.streamlit.io/ and sign in with GitHub.
5. Select **Create app**, choose the new repository, branch `main`, and main file path `app.py`.
6. Deploy.

## Data endpoint
`GET https://api.mexc.com/api/v1/contract/kline/BTC_USDT?interval=Min1`

MEXC documents the futures candle endpoint as `/api/v1/contract/kline/{symbol}` and lists `Min1` as a supported interval. Public endpoint availability can vary; the app shows the API error if it cannot fetch data.

## Limitations
- Free Streamlit hosting may sleep/restart and is not guaranteed to run continuously 24/7.
- Auto-refresh runs while the app session is active; it is not a background always-on worker.
- Forecasts are statistical estimates, not guaranteed predictions or financial advice.
- Evaluation uses a chronological split and should not be treated as evidence of profitability.
