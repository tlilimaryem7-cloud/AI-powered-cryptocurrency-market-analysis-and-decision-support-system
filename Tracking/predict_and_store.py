# ============================================================
# STEP 3 — Prediction Pipeline → predictions table
# ============================================================
# Loads saved BTC and ETH models, runs prediction on today's
# live features, and inserts into the predictions table.
#
# prediction_date = today   (when we run the model)
# target_date     = tomorrow (when the true price will be known)
#
# Run daily AFTER daily_fetch.py
# Run: python predict_and_store.py
# ============================================================

import sys
import os
import joblib
import psycopg2
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta

# ─────────────────────────────────────────────────────────────
# PATH SETUP
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = r"C:\Users\tlili\OneDrive\Bureau\Bootcamp\AI-powered-cryptocurrency-market-analysis-and-decision-support-system"
SRC_PATH     = os.path.join(PROJECT_ROOT, "src")
sys.path.append(PROJECT_ROOT)
sys.path.append(SRC_PATH)

# ─────────────────────────────────────────────────────────────
# DATABASE CONFIG
# ─────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host"    : "localhost",
    "port"    : 5432,
    "user"    : "postgres",
    "password": "Mt889933!!",
    "dbname"  : "crypto_tracker",
}

# ─────────────────────────────────────────────────────────────
# MODEL PATHS
# ─────────────────────────────────────────────────────────────
MODEL_PATHS = {
    "btc": os.path.join(PROJECT_ROOT, "models", "saved_models", "btc_model_v2.pkl"),
    "eth": os.path.join(PROJECT_ROOT, "models", "saved_models", "eth_model_v2.pkl"),
}

MODEL_VERSION = "v2"

# ─────────────────────────────────────────────────────────────
# FEATURES — must match exactly what the models were trained on
# ─────────────────────────────────────────────────────────────
# BTC — 20 SHAP-selected features (v2)
FEATURES_BTC = [
    "price_to_ma7", "rsi_14_lag1", "rsi_14", "bb_pct",
    "macd_histogram", "momentum_acceleration", "price_lag1",
    "spy_return", "rsi_14_lag3", "fear_greed_lag7",
    "spy_return_ma7", "fear_greed_lag1", "volume_lag1",
    "macd_lag1", "dxy_return_ma7", "macd_lag3",
    "rsi_14_lag2", "price_lag3", "price_to_ma30", "volatility_7d",
]

# ETH — 20 SHAP-selected features (v2)
FEATURES_ETH = [
    "rsi_14_lag1", "price_to_ma7", "rsi_14", "macd_histogram",
    "bb_pct", "rsi_14_lag3", "momentum_acceleration", "spy_return",
    "fear_greed_lag1", "macd_lag1", "fear_greed_lag7", "macd_lag3",
    "spy_return_ma7", "volatility_21d", "price_lag1", "price_to_ma30",
    "volatility_21d_lag3", "price_lag3", "volatility_7d", "vix_ma14",
]

FEATURES_MAP = {"btc": FEATURES_BTC, "eth": FEATURES_ETH}

TODAY     = date.today()
TOMORROW  = TODAY + timedelta(days=1)
COINS     = ["btc", "eth"]
TICKERS   = {"btc": "BTC-USD", "eth": "ETH-USD"}


# ─────────────────────────────────────────────────────────────
# BLOCK 1 — BUILD LIVE FEATURES
# ─────────────────────────────────────────────────────────────
def fetch_fear_greed(limit=30):
    url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()["data"]
        df = pd.DataFrame(data)[["timestamp", "value"]]
        df["timestamp"]  = pd.to_datetime(df["timestamp"].astype(int), unit="s")
        df["timestamp"]  = df["timestamp"].dt.normalize()
        df["fear_greed"] = df["value"].astype(float)
        df = df[["timestamp", "fear_greed"]].sort_values("timestamp").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"  ⚠️  Fear & Greed fetch failed: {e}")
        return None


def build_live_features(coin: str) -> pd.Series:
    """
    Fetch the last 100 days of data and compute all 21 features.
    Returns the latest row as a pd.Series.
    """
    ticker = TICKERS[coin]

    # ── Crypto prices
    df = yf.download(ticker, period="120d", interval="1d",
                     auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"❌ No data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index().rename(columns={"Date": "timestamp",
                                           "Close": "price",
                                           "Volume": "volume"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── Macro signals
    for name, yticker, col in [
        ("spy_return", "SPY",      "spy_close"),
        ("dxy_return", "DX-Y.NYB", "dxy_close"),
        ("vix",        "^VIX",     "vix"),
    ]:
        m = yf.download(yticker, period="120d", interval="1d",
                        auto_adjust=True, progress=False)
        if m.empty:
            df[name] = np.nan
            continue
        if isinstance(m.columns, pd.MultiIndex):
            m.columns = m.columns.get_level_values(0)
        m = m.reset_index().rename(columns={"Date": "timestamp", "Close": col})
        m["timestamp"] = pd.to_datetime(m["timestamp"])
        if name == "vix":
            m["vix"] = m[col]
            df = df.merge(m[["timestamp", "vix"]], on="timestamp", how="left")
        else:
            m[name] = np.log(m[col] / m[col].shift(1))
            df = df.merge(m[["timestamp", name]], on="timestamp", how="left")

    # ── Fear & Greed
    fg = fetch_fear_greed(limit=60)
    if fg is not None:
        df = df.merge(fg, on="timestamp", how="left")
        df["fear_greed"] = df["fear_greed"].ffill()
    else:
        df["fear_greed"] = 50.0

    # ── Forward fill macro
    for col in ["spy_return", "dxy_return", "vix"]:
        df[col] = df[col].ffill(limit=3)

    # ── Feature engineering
    df["log_return_1d"]    = np.log(df["price"] / df["price"].shift(1))
    df["volatility_7d"]    = df["log_return_1d"].rolling(7).std()
    df["volatility_21d"]   = df["log_return_1d"].rolling(21).std()

    # RSI
    delta    = df["price"].diff()
    gain     = delta.clip(lower=0).rolling(14).mean()
    loss     = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi_14"] = 100 - (100 / (1 + gain / loss))

    # Moving averages
    for w in [7, 14, 30, 50]:
        df[f"ma_{w}"] = df["price"].rolling(w).mean()

    df["price_to_ma7"]  = (df["price"] - df["ma_7"])  / df["ma_7"]
    df["price_to_ma30"] = (df["price"] - df["ma_30"]) / df["ma_30"]
    df["price_to_ma50"] = (df["price"] - df["ma_50"]) / df["ma_50"]

    # MACD
    ema12 = df["price"].ewm(span=12, adjust=False).mean()
    ema26 = df["price"].ewm(span=26, adjust=False).mean()
    df["macd"]           = ema12 - ema26
    df["macd_signal"]    = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]

    # Momentum
    df["momentum_5d"]           = df["price"].pct_change(5)
    df["momentum_10d"]          = df["price"].pct_change(10)
    df["momentum_acceleration"] = df["momentum_5d"] - df["momentum_10d"]

    # Bollinger Bands
    bb_std         = df["price"].rolling(14).std()
    df["bb_width"] = (4 * bb_std) / df["ma_14"]
    df["bb_pct"]   = (df["price"] - (df["ma_14"] - 2 * bb_std)) / (4 * bb_std)

    # Macro rolling
    df["spy_return_ma7"]  = df["spy_return"].rolling(7).mean()
    df["spy_return_std7"] = df["spy_return"].rolling(7).std()
    df["dxy_return_ma7"]  = df["dxy_return"].rolling(7).mean()
    df["vix_ma14"]        = df["vix"].rolling(14).mean()
    df["vix_regime"]      = (df["vix"] > 20).astype(int)

    # Sentiment
    df["fear_greed_ma7"]  = df["fear_greed"].rolling(7).mean()
    df["fear_greed_lag1"] = df["fear_greed"].shift(1)
    df["fear_greed_lag7"] = df["fear_greed"].shift(7)

    # Regime flags
    df["bull_bear_flag"]    = (df["price"] > df["ma_50"]).astype(int)
    vol_median              = df["volatility_21d"].rolling(90).median()
    df["volatility_regime"] = (df["volatility_21d"] > vol_median).astype(int)

    # Lag features (required by v2 models)
    for col, lags in [
        ("rsi_14",         [1, 2, 3]),
        ("macd",           [1, 3]),
        ("volume",         [1]),
        ("price",          [1, 3]),
        ("volatility_21d", [3]),
    ]:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # Drop NaNs and return latest row
    features_list = FEATURES_MAP[coin]
    df = df.dropna(subset=features_list).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"❌ No valid rows after feature engineering for {coin}")

    latest = df.iloc[-1]
    print(f"  Feature date : {latest['timestamp'].date()}")
    return latest


# ─────────────────────────────────────────────────────────────
# BLOCK 2 — RUN PREDICTION
# ─────────────────────────────────────────────────────────────
def predict(coin: str, model) -> dict:
    """Build live features and run model prediction."""
    print(f"\n  Building live features for {coin.upper()}...")
    latest = build_live_features(coin)

    features_list = FEATURES_MAP[coin]
    X = pd.DataFrame([latest[features_list]])

    prob       = model.predict_proba(X)[0]
    pred_class = int(model.predict(X)[0])
    confidence = round(float(max(prob)) * 100, 2)
    direction  = "UP" if pred_class == 1 else "DOWN"

    print(f"  Direction  : {direction}")
    print(f"  Confidence : {confidence}%")
    print(f"  Prob UP    : {round(prob[1]*100, 2)}% | Prob DOWN: {round(prob[0]*100, 2)}%")

    return {
        "direction" : direction,
        "confidence": confidence,
        "prob_up"   : round(float(prob[1]), 4),
        "prob_down" : round(float(prob[0]), 4),
    }


# ─────────────────────────────────────────────────────────────
# BLOCK 3 — INSERT INTO predictions table
# ─────────────────────────────────────────────────────────────
def insert_prediction(conn, coin: str, pred: dict):
    """Insert prediction into predictions table. Skip if already exists."""
    cur = conn.cursor()

    cur.execute(
        "SELECT id FROM predictions WHERE coin = %s AND prediction_date = %s",
        (coin, TODAY)
    )
    if cur.fetchone():
        print(f"  ℹ️  {coin.upper()} — prediction already exists for {TODAY}, skipping")
        cur.close()
        return False

    cur.execute("""
        INSERT INTO predictions
            (coin, prediction_date, target_date,
             predicted_direction, confidence, model_version)
        VALUES
            (%s, %s, %s, %s, %s, %s)
    """, (
        coin,
        TODAY,
        TOMORROW,
        pred["direction"],
        pred["confidence"],
        MODEL_VERSION,
    ))

    conn.commit()
    cur.close()
    return True


# ─────────────────────────────────────────────────────────────
# BLOCK 4 — DISPLAY SUMMARY
# ─────────────────────────────────────────────────────────────
def show_summary(conn):
    """Print current state of predictions table."""
    cur = conn.cursor()
    cur.execute("""
        SELECT coin, prediction_date, target_date,
               predicted_direction, confidence, model_version
        FROM predictions
        ORDER BY prediction_date DESC, coin
        LIMIT 10
    """)
    rows = cur.fetchall()
    print(f"\n  {'Coin':<6} {'Pred Date':<12} {'Target':<12} "
          f"{'Direction':<10} {'Confidence':>10} {'Model':>6}")
    print(f"  {'-'*60}")
    for row in rows:
        print(f"  {row[0]:<6} {str(row[1]):<12} {str(row[2]):<12} "
              f"{row[3]:<10} {float(row[4]):>9.2f}% {row[5]:>6}")
    cur.close()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print(f"  STEP 3 — Predict → predictions table")
    print(f"  Prediction date : {TODAY}")
    print(f"  Target date     : {TOMORROW}")
    print("="*55)

    # Connect to DB
    conn = psycopg2.connect(**DB_CONFIG)
    print("  ✅ Connected to crypto_tracker")

    for coin in COINS:
        print(f"\n{'─'*55}")
        print(f"  🔮 {coin.upper()} PREDICTION")
        print(f"{'─'*55}")

        try:
            # Load model
            model_path = MODEL_PATHS[coin]
            print(f"  Loading model from: {model_path}")
            model = joblib.load(model_path)
            print(f"  ✅ Model loaded")

            # Run prediction
            pred = predict(coin, model)

            # Insert into DB
            inserted = insert_prediction(conn, coin, pred)
            if inserted:
                print(f"  ✅ {coin.upper()} prediction inserted into predictions table")

        except Exception as e:
            print(f"  ❌ {coin.upper()} failed: {e}")

    # Summary
    print("\n📊 predictions table (last 10 rows):")
    show_summary(conn)

    conn.close()

    print("\n" + "="*55)
    print("  ✅ Step 3 complete!")
    print("  Next: run error_tracker.py (Step 4)")
    print("="*55)