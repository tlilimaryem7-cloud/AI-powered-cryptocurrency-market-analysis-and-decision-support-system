# ============================================================
# LIVE PIPELINE — BTC/ETH LIVE FEATURE ENGINEERING
# ============================================================
# Input  : fetches last 120 days of live data (warmup window)
# Output : single row of 21 features ready for model prediction
#
# Warmup rationale:
#   - ma_50           needs 50 days
#   - volatility_21d  needs 21 days
#   - vix_ma14        needs 14 days
#   - volatility_regime rolling(90) needs 90 days
#   - 120 days = safe buffer for all rolling windows
#
# Run    : python live_pipeline.py --coin btc
#          python live_pipeline.py --coin eth
# ============================================================

import yfinance  as yf
import pandas    as pd
import numpy     as np
import requests
import argparse
import joblib
import os
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────
WARMUP_DAYS = 120          # days of history needed for rolling features
MODELS_PATH = r"C:\Users\tlili\OneDrive\Bureau\Bootcamp\AI-powered-cryptocurrency-market-analysis-and-decision-support-system\models\saved_models"
FEATURES = [
    # Trend
    "price_to_ma7", "price_to_ma30", "price_to_ma50",
    # Momentum
    "rsi_14", "macd_histogram", "momentum_acceleration",
    # Volatility
    "volatility_7d", "volatility_21d", "bb_width", "bb_pct",
    # Macro
    "spy_return", "spy_return_ma7", "spy_return_std7",
    "dxy_return_ma7", "vix_ma14", "vix_regime",
    # Sentiment
    "fear_greed_ma7", "fear_greed_lag1", "fear_greed_lag7",
    # Regime flags
    "bull_bear_flag", "volatility_regime",
]

# VIX outlier caps (from training data — must stay fixed)
VIX_CHANGE_P01 = -4.65
VIX_CHANGE_P99 =  5.73


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def get_date_range():
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")
    return start, end


def download_yf(ticker, start, end, col_rename):
    df = yf.download(ticker, start=start, end=end,
                     interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        print(f"  ⚠️  {ticker} returned empty!")
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()[["Date", "Close"]].rename(
        columns={"Date": "timestamp", "Close": col_rename}
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"  ✅ {ticker}: {len(df)} rows | "
          f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    return df


def add_log_return(df, close_col, return_col):
    df = df.sort_values("timestamp").copy()
    df[return_col] = np.log(df[close_col] / df[close_col].shift(1))
    return df.drop(columns=[close_col])


def fetch_fear_greed(limit=150):
    url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()["data"]
        df = pd.DataFrame(data)[["timestamp", "value"]]
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
        df["timestamp"] = df["timestamp"].dt.normalize()
        df["fear_greed"] = df["value"].astype(float)
        df = df[["timestamp", "fear_greed"]].sort_values("timestamp").reset_index(drop=True)
        print(f"  ✅ Fear & Greed: {len(df)} rows | "
              f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
        return df
    except Exception as e:
        print(f"  ❌ Fear & Greed fetch failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────
def build_live_features(coin: str) -> pd.Series:
    """
    Fetches last 120 days of data for a given coin,
    computes all rolling features, returns the last row
    as a Series of exactly 21 features ready for prediction.

    Parameters
    ----------
    coin : str — "btc" or "eth"

    Returns
    -------
    pd.Series — 21 features, indexed by feature name
    """

    ticker = f"{coin.upper()}-USD"
    start, end = get_date_range()

    print(f"\n{'='*55}")
    print(f"  LIVE PIPELINE — {coin.upper()}")
    print(f"  Warmup window : {start} → {end}")
    print(f"{'='*55}")

    # ── BLOCK 1 : Crypto prices
    print("\n📥 Fetching crypto prices...")
    df_raw = yf.download(ticker, start=start, end=end,
                         interval="1d", auto_adjust=True, progress=False)
    if df_raw.empty:
        raise ValueError(f"❌ No data returned for {ticker}")
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    df_raw = df_raw.reset_index()
    df_raw = df_raw.rename(columns={"Date": "timestamp",
                                    "Close": "price",
                                    "Volume": "volume"})
    df_raw = df_raw[["timestamp", "price", "volume"]]
    df_raw["coin"]      = coin.lower()
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
    print(f"  ✅ {ticker}: {len(df_raw)} rows")

    # ── BLOCK 2 : Macro signals
    print("\n📥 Fetching macro signals...")
    spy_df = download_yf("SPY",      start, end, "spy_close")
    dxy_df = download_yf("DX-Y.NYB", start, end, "dxy_close")
    vix_df = download_yf("^VIX",     start, end, "vix")

    # ── BLOCK 3 : Fear & Greed
    print("\n📥 Fetching Fear & Greed...")
    fg_df = fetch_fear_greed(limit=150)

    # ── BLOCK 4 : Macro returns
    print("\n⚙️  Computing macro returns...")
    if spy_df is not None:
        spy_df = add_log_return(spy_df, "spy_close", "spy_return")
    if dxy_df is not None:
        dxy_df = add_log_return(dxy_df, "dxy_close", "dxy_return")
    if vix_df is not None:
        vix_df["vix_change"] = vix_df["vix"].diff()

    # ── BLOCK 5 : Merge
    print("\n⚙️  Merging signals...")
    df = df_raw.copy()
    for name, ext_df in [("SPY", spy_df), ("DXY", dxy_df),
                          ("VIX", vix_df), ("Fear&Greed", fg_df)]:
        if ext_df is None:
            print(f"  ⚠️  {name} skipped")
            continue
        df = df.merge(ext_df, on="timestamp", how="left")
    print(f"  Shape after merge: {df.shape}")

    # ── BLOCK 6 : Preprocessing
    print("\n⚙️  Preprocessing...")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Rounding
    df["price"]      = df["price"].round(2)
    df["volume"]     = df["volume"].round(0).astype(float)
    df["spy_return"] = df["spy_return"].round(6)
    df["dxy_return"] = df["dxy_return"].round(6)
    df["vix"]        = df["vix"].round(2)
    df["vix_change"] = df["vix_change"].round(2)
    df["fear_greed"] = df["fear_greed"].round(0)

    # Forward-fill macro (weekends/holidays)
    macro_cols = ["spy_return", "dxy_return", "vix", "vix_change"]
    df[macro_cols] = df[macro_cols].ffill(limit=3)

    # Forward-fill fear & greed
    df["fear_greed"] = df["fear_greed"].ffill()

    # Cap vix_change — use TRAINING caps (never recompute from live data)
    df["vix_change"] = df["vix_change"].clip(
        lower=VIX_CHANGE_P01, upper=VIX_CHANGE_P99
    )

    # ── BLOCK 7 : Feature engineering
    print("\n⚙️  Computing features...")

    # Log return
    df["log_return_1d"] = np.log(df["price"] / df["price"].shift(1))

    # Volatility
    df["volatility_7d"]  = df["log_return_1d"].rolling(7).std()
    df["volatility_14d"] = df["log_return_1d"].rolling(14).std()
    df["volatility_21d"] = df["log_return_1d"].rolling(21).std()

    # RSI
    delta    = df["price"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    df["rsi_14"] = 100 - (100 / (1 + avg_gain / avg_loss))

    # Moving averages
    for w in [7, 14, 30, 50]:
        df[f"ma_{w}"] = df["price"].rolling(w).mean()

    df["price_to_ma7"]  = (df["price"] - df["ma_7"])  / df["ma_7"]
    df["price_to_ma30"] = (df["price"] - df["ma_30"]) / df["ma_30"]
    df["price_to_ma50"] = (df["price"] - df["ma_50"]) / df["ma_50"]

    # MACD
    ema_12             = df["price"].ewm(span=12, adjust=False).mean()
    ema_26             = df["price"].ewm(span=26, adjust=False).mean()
    df["macd"]         = ema_12 - ema_26
    df["macd_signal"]  = df["macd"].ewm(span=9, adjust=False).mean()
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

    print("  └─ Feature engineering complete ✅")

    # ── BLOCK 8 : Extract last row (today's features)
    last_row = df.dropna(subset=FEATURES).iloc[-1]

    # Verify all 21 features present
    missing = [f for f in FEATURES if f not in last_row.index or pd.isna(last_row[f])]
    if missing:
        raise ValueError(f"❌ Missing features: {missing}")

    print(f"\n  Date        : {last_row['timestamp'].date()}")
    print(f"  Price       : ${last_row['price']:,.2f}")
    print(f"  Features    : {len(FEATURES)} ✅")
    print(f"  NaN check   : {sum(pd.isna(last_row[f]) for f in FEATURES)} (should be 0)")

    return last_row[FEATURES]


# ─────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────
def predict(coin: str) -> dict:
    """
    Full pipeline: fetch → features → predict → return result.

    Returns
    -------
    dict with keys:
        coin, date, price, direction, confidence, model
    """
    coin = coin.lower()
    if coin not in ["btc", "eth"]:
        raise ValueError("coin must be 'btc' or 'eth'")

    # Build features
    features = build_live_features(coin)

    # Load model
    model_path = os.path.join(MODELS_PATH, f"{coin}_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found: {model_path}")
    model = joblib.load(model_path)

    # Predict
    X = pd.DataFrame([features], columns=FEATURES)
    direction  = model.predict(X)[0]
    confidence = model.predict_proba(X)[0][direction]

    result = {
        "coin"      : coin.upper(),
        "date"      : datetime.today().strftime("%Y-%m-%d"),
        "direction" : "UP 📈" if direction == 1 else "DOWN 📉",
        "confidence": round(confidence * 100, 2),
        "model"     : "Stacking (RF+GB+XGB→LR)" if coin == "btc"
                      else "GradientBoosting (tuned)",
    }

    print(f"\n{'='*55}")
    print(f"  PREDICTION — {result['coin']}")
    print(f"{'='*55}")
    print(f"  Date       : {result['date']}")
    print(f"  Direction  : {result['direction']}")
    print(f"  Confidence : {result['confidence']}%")
    print(f"  Model      : {result['model']}")
    print(f"{'='*55}")

    return result


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Live crypto direction prediction"
    )
    parser.add_argument(
        "--coin", type=str, required=True,
        choices=["btc", "eth"],
        help="Coin to predict: btc or eth"
    )
    args   = parser.parse_args()
    result = predict(args.coin)
