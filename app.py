import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, brier_score_loss

st.set_page_config(page_title="BTC Futures Lab", page_icon="📈", layout="wide")
st.title("BTC Futures Lab")
st.caption("Read-only BTC_USDT futures analysis using public MEXC market data. No trading keys or order placement.")

BASE_URL = "https://api.mexc.com"
KLINE_PATH = "/api/v1/contract/kline/BTC_USDT"

@st.cache_resource
def make_session():
    session = requests.Session()
    retry = Retry(
        total=5, connect=5, read=5, backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "BTC-Futures-Lab/1.0"})
    return session

def fetch_candles():
    response = make_session().get(
        BASE_URL + KLINE_PATH,
        params={"interval": "Min1"},
        timeout=(8, 20),
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success", False):
        raise RuntimeError(f"MEXC API returned success=false: {payload.get('message', 'unknown response')}")
    data = payload.get("data")
    if not isinstance(data, dict) or "time" not in data or "close" not in data:
        raise RuntimeError("Unexpected MEXC candle response format. Check the API documentation or response.")
    df = pd.DataFrame({
        "time": data.get("time", []),
        "open": data.get("open", []),
        "high": data.get("high", []),
        "low": data.get("low", []),
        "close": data.get("close", []),
        "volume": data.get("vol", data.get("volume", [])),
    })
    if df.empty:
        raise RuntimeError("MEXC returned no candle rows.")
    df["time"] = pd.to_datetime(pd.to_numeric(df["time"], errors="coerce"), unit="s", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["time", "close"]).sort_values("time").drop_duplicates("time")
    # Exclude the most recent candle because it may still be forming.
    if len(df) > 1:
        df = df.iloc[:-1].copy()
    return df.reset_index(drop=True)

def build_features(df):
    x = pd.DataFrame(index=df.index)
    close = df["close"]
    ret = np.log(close).diff()
    x["ret_1"] = ret
    x["ret_3"] = np.log(close / close.shift(3))
    x["ret_5"] = np.log(close / close.shift(5))
    x["ret_10"] = np.log(close / close.shift(10))
    x["volatility_10"] = ret.rolling(10).std()
    x["volatility_30"] = ret.rolling(30).std()
    x["volume_ratio"] = df["volume"] / df["volume"].rolling(30).mean()
    x["range_pct"] = (df["high"] - df["low"]) / close
    return x.replace([np.inf, -np.inf], np.nan)

def prepare_dataset(df):
    features = build_features(df)
    # Label at candle t: whether close five candles later is higher than close at t.
    future_close = df["close"].shift(-5)
    y = (future_close > df["close"]).astype(int)
    data = features.copy()
    data["target"] = y
    data["time"] = df["time"]
    data["close"] = df["close"]
    data = data.dropna().reset_index(drop=True)
    # The final five labels are unknown, so remove them explicitly.
    if len(data) > 5:
        data = data.iloc[:-5].copy()
    return data, features

def train_and_evaluate(data):
    feature_cols = [c for c in data.columns if c not in ("target", "time", "close")]
    n = len(data)
    if n < 150:
        raise RuntimeError(f"Only {n} usable labelled observations. Need at least 150; fetch more history.")
    split = int(n * 0.70)
    train = data.iloc[:split]
    test = data.iloc[split:]
    model = Pipeline([
        ("scale", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000, class_weight=None, random_state=42)),
    ])
    model.fit(train[feature_cols], train["target"])
    probs = model.predict_proba(test[feature_cols])[:, 1]
    preds = (probs >= 0.5).astype(int)
    accuracy = accuracy_score(test["target"], preds)
    baseline_class = int(train["target"].mean() >= 0.5)
    baseline_acc = accuracy_score(test["target"], np.full(len(test), baseline_class))
    brier = brier_score_loss(test["target"], probs)
    # Test-period prevalence is descriptive only, not a deployable forecast baseline.
    train_rate = float(train["target"].mean())
    constant_brier = brier_score_loss(test["target"], np.full(len(test), train_rate))
    return model, feature_cols, {
        "test_n": len(test),
        "accuracy": accuracy,
        "majority_baseline": baseline_acc,
        "brier": brier,
        "constant_brier": constant_brier,
        "test_up_rate": float(test["target"].mean()),
        "train_end": train["time"].iloc[-1],
        "test_start": test["time"].iloc[0],
    }

with st.sidebar:
    st.header("Settings")
    refresh_seconds = st.selectbox("Auto-refresh interval", [15, 30, 60, 120], index=2)
    st.caption("Refresh works while the dashboard session is active. Free hosting may sleep.")

try:
    with st.spinner("Fetching public MEXC futures candles..."):
        df = fetch_candles()
    if len(df) < 150:
        st.warning(f"Only {len(df)} closed candles received. More history is needed for a meaningful evaluation.")
    st.caption(f"Last successful data fetch: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} · Closed candles: {len(df)}")

    if len(df):
        latest = df.iloc[-1]
        col1, col2, col3 = st.columns(3)
        col1.metric("BTC_USDT last closed price", f"${latest['close']:,.2f}")
        col2.metric("Latest closed candle (UTC)", latest["time"].strftime("%Y-%m-%d %H:%M"))
        col3.metric("Candles received", f"{len(df):,}")

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df["time"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name="BTC_USDT"
        ))
        fig.update_layout(height=420, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=25, b=10))
        st.plotly_chart(fig, use_container_width=True)

        data, all_features = prepare_dataset(df)
        try:
            model, feature_cols, metrics = train_and_evaluate(data)
            latest_features = all_features.iloc[[-1]][feature_cols].replace([np.inf, -np.inf], np.nan)
            if latest_features.isna().any(axis=None):
                st.info("The latest candle does not yet have all features. Refresh after more candles arrive.")
            else:
                p_up = float(model.predict_proba(latest_features)[:, 1][0])
                p_down = 1.0 - p_up
                st.subheader("Five-minute statistical estimate")
                p1, p2, p3 = st.columns(3)
                p1.metric("Estimated UP probability", f"{p_up:.1%}")
                p2.metric("Estimated DOWN probability", f"{p_down:.1%}")
                margin = abs(p_up - 0.5) * 2
                p3.metric("Distance from 50/50", f"{margin:.1%}")
                if margin < 0.10:
                    st.warning("Low-confidence estimate: probability is close to 50/50. Treat as no clear directional edge.")
                elif p_up > 0.5:
                    st.info("Model leans UP, but this is not a reliable trading signal by itself.")
                else:
                    st.info("Model leans DOWN, but this is not a reliable trading signal by itself.")

            st.subheader("Chronological out-of-sample evaluation")
            m1, m2, m3 = st.columns(3)
            m1.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
            m2.metric("Majority-class baseline", f"{metrics['majority_baseline']:.1%}")
            m3.metric("Brier score (lower is better)", f"{metrics['brier']:.4f}")
            st.write({
                "test observations": metrics["test_n"],
                "test UP frequency": round(metrics["test_up_rate"], 4),
                "constant training-rate Brier baseline": round(metrics["constant_brier"], 4),
                "training ended (UTC)": str(metrics["train_end"]),
                "test began (UTC)": str(metrics["test_start"]),
            })
            if metrics["accuracy"] <= metrics["majority_baseline"] or metrics["brier"] >= metrics["constant_brier"]:
                st.warning("Current model has not beaten all simple baselines on this test period. Do not assume predictive edge.")
            st.caption("Evaluation is a simple chronological 70/30 split, not proof of future profitability. No fees, slippage, funding or event settlement are simulated.")
        except Exception as model_error:
            st.error(f"Model/evaluation could not run: {model_error}")
            st.info("The live market chart may still work. More valid historical candles may be needed.")
    else:
        st.error("No closed candles were returned.")
except Exception as exc:
    st.error("Could not retrieve MEXC public futures candles.")
    st.code(f"{type(exc).__name__}: {exc}")
    st.caption("The app retries temporary HTTP errors. If the issue persists, check MEXC API availability and the endpoint documentation.")

st.divider()
st.caption("Read-only research tool. No exchange account access, API keys, order placement or financial guarantees.")

# Lightweight refresh loop; Streamlit reruns the script after the delay.
time.sleep(refresh_seconds)
st.rerun()
