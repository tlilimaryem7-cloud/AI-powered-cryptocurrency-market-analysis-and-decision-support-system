# ============================================================
# STEP 6 — Retraining Script
# ============================================================
# Checks retraining_log for pending alerts, fetches fresh data,
# retrains the model, evaluates it, and saves it if improved.
#
# Run: python retrain.py
# Run: python retrain.py --force btc   (force retrain regardless of alert)
# ============================================================

import sys
import os
import joblib
import psycopg2
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import argparse
from datetime import date, datetime, timedelta

# ─────────────────────────────────────────────────────────────
# PATH SETUP
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = r"C:\Users\tlili\OneDrive\Bureau\Bootcamp\AI-powered-cryptocurrency-market-analysis-and-decision-support-system"
sys.path.append(PROJECT_ROOT)

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

BACKUP_DIR = os.path.join(PROJECT_ROOT, "models", "saved_models", "backups")
os.makedirs(BACKUP_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# FEATURES — same as training
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

# All features needed for engineering (superset of both models)
ALL_FEATURES = list(set(FEATURES_BTC + FEATURES_ETH))

TICKERS = {"btc": "BTC-USD", "eth": "ETH-USD"}

# ─────────────────────────────────────────────────────────────
# IMPORTS — ML models
# ─────────────────────────────────────────────────────────────
from sklearn.ensemble         import GradientBoostingClassifier
from sklearn.metrics          import accuracy_score, matthews_corrcoef


# ═══════════════════════════════════════════════════════════════
# BLOCK 1 — CHECK PENDING ALERTS
# ═══════════════════════════════════════════════════════════════
def get_pending_alerts(conn) -> list:
    """Return list of coins with pending retraining alerts."""
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT coin, triggered_at, trigger_reason,
                        rolling_accuracy_before
        FROM retraining_log
        WHERE retrain_status = 'pending'
        ORDER BY triggered_at ASC
    """)
    rows = cur.fetchall()
    cur.close()
    return rows


# ═══════════════════════════════════════════════════════════════
# BLOCK 2 — FETCH FRESH TRAINING DATA
# ═══════════════════════════════════════════════════════════════
def fetch_fear_greed(limit=3000):
    url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()["data"]
        df = pd.DataFrame(data)[["timestamp", "value"]]
        df["timestamp"]  = pd.to_datetime(df["timestamp"].astype(int), unit="s")
        df["timestamp"]  = df["timestamp"].dt.normalize()
        df["fear_greed"] = df["value"].astype(float)
        return df[["timestamp", "fear_greed"]].sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        print(f"  ⚠️  Fear & Greed failed: {e}")
        return None


def fetch_training_data(coin: str,
                        start: str = "2017-01-01") -> pd.DataFrame:
    """
    Fetch full historical data + macro signals + features.
    Same logic as the original pipeline.py — always fetches up to today.
    """
    end = date.today().strftime("%Y-%m-%d")
    print(f"  Fetching {coin.upper()} data: {start} → {end}")

    ticker = TICKERS[coin]

    # Crypto prices
    df = yf.download(ticker, start=start, end=end,
                     interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index().rename(columns={
        "Date": "timestamp", "Close": "price", "Volume": "volume"
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["coin"]      = coin
    df = df[["timestamp", "coin", "price", "volume"]].sort_values("timestamp")

    # Macro signals
    for name, yticker, col in [
        ("spy_return", "SPY",      "spy_close"),
        ("dxy_return", "DX-Y.NYB", "dxy_close"),
        ("vix",        "^VIX",     "vix"),
    ]:
        m = yf.download(yticker, start=start, end=end,
                        interval="1d", auto_adjust=True, progress=False)
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

    # Fear & Greed
    fg = fetch_fear_greed()
    if fg is not None:
        df = df.merge(fg, on="timestamp", how="left")
        df["fear_greed"] = df["fear_greed"].ffill()
    else:
        df["fear_greed"] = 50.0

    # Forward fill macro
    for col in ["spy_return", "dxy_return", "vix"]:
        df[col] = df[col].ffill(limit=3)

    print(f"  ✅ Raw data shape: {df.shape}")
    return df


# ═══════════════════════════════════════════════════════════════
# BLOCK 3 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 21 features — same logic as pipeline.py."""
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["log_return_1d"]  = np.log(df["price"] / df["price"].shift(1))
    df["volatility_7d"]  = df["log_return_1d"].rolling(7).std()
    df["volatility_21d"] = df["log_return_1d"].rolling(21).std()

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
    df["bb_pct"]   = (df["price"] - (df["ma_14"] - 2*bb_std)) / (4*bb_std)

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

    # Target
    df["target"] = (df["log_return_1d"] > 0).astype(int)

    df = df.dropna(subset=ALL_FEATURES + ["target"]).reset_index(drop=True)
    print(f"  ✅ Features built — {len(df)} rows")
    return df


# ═══════════════════════════════════════════════════════════════
# BLOCK 4 — BUILD NEW MODEL
# ═══════════════════════════════════════════════════════════════
def build_model(coin: str):
    """Return a fresh untrained v2 model with Optuna-tuned hyperparameters."""
    if coin == "btc":
        return GradientBoostingClassifier(
            n_estimators=349, max_depth=4,
            learning_rate=0.01438, min_samples_leaf=75,
            subsample=0.621, random_state=42
        )
    else:  # eth
        return GradientBoostingClassifier(
            n_estimators=437, max_depth=6,
            learning_rate=0.01777, min_samples_leaf=97,
            subsample=0.507, random_state=42
        )


# ═══════════════════════════════════════════════════════════════
# BLOCK 5 — TRAIN + EVALUATE
# ═══════════════════════════════════════════════════════════════
def train_and_evaluate(coin: str, df: pd.DataFrame) -> tuple:
    """
    Split data using S2 regime-aware split,
    train new model, evaluate on test set.
    Returns (new_model, test_accuracy, train_end, test_start)
    """
    dates = df["timestamp"]

    # S2 Regime-Aware Split — extended to include latest data
    train = df[dates < "2023-01-01"]
    val   = df[(dates >= "2023-01-01") & (dates < "2025-01-01")]
    test  = df[dates >= "2025-01-01"]

    print(f"  Train : {train['timestamp'].min().date()} → "
          f"{train['timestamp'].max().date()} ({len(train)} rows)")
    print(f"  Val   : {val['timestamp'].min().date()} → "
          f"{val['timestamp'].max().date()} ({len(val)} rows)")
    print(f"  Test  : {test['timestamp'].min().date()} → "
          f"{test['timestamp'].max().date()} ({len(test)} rows)")

    if len(train) < 100 or len(test) < 10:
        raise ValueError("Not enough data to retrain")

    # Combine train + val for final training (standard practice)
    train_full = pd.concat([train, val], ignore_index=True)

    features_list = FEATURES_MAP[coin]

    model = build_model(coin)
    print(f"\n  Training new {coin.upper()} model...")
    model.fit(train_full[features_list], train_full["target"])

    # Evaluate on test set
    test_preds    = model.predict(test[features_list])
    test_accuracy = round(accuracy_score(test["target"], test_preds), 4)
    test_mcc      = round(matthews_corrcoef(test["target"], test_preds), 4)
    print(f"  ✅ New model test accuracy : {test_accuracy*100:.1f}%")
    print(f"  ✅ New model MCC           : {test_mcc:.4f}")

    return (
        model,
        test_accuracy,
        train["timestamp"].min().date(),
        test["timestamp"].max().date(),
    )


# ═══════════════════════════════════════════════════════════════
# BLOCK 6 — EVALUATE OLD MODEL
# ═══════════════════════════════════════════════════════════════
def evaluate_old_model(coin: str, df: pd.DataFrame) -> float:
    """Load old v2 model and evaluate on the test set."""
    try:
        old_model     = joblib.load(MODEL_PATHS[coin])
        features_list = FEATURES_MAP[coin]
        dates         = df["timestamp"]
        test          = df[dates >= "2025-01-01"]
        if test.empty:
            return 0.0
        preds    = old_model.predict(test[features_list])
        accuracy = round(accuracy_score(test["target"], preds), 4)
        print(f"  Old model test accuracy : {accuracy*100:.1f}%")
        return accuracy
    except Exception as e:
        print(f"  ⚠️  Could not evaluate old model: {e}")
        return 0.0


# ═══════════════════════════════════════════════════════════════
# BLOCK 7 — SAVE NEW MODEL + BACKUP OLD
# ═══════════════════════════════════════════════════════════════
def save_model(coin: str, new_model, version: str):
    """Backup old model and save new one."""
    model_path = MODEL_PATHS[coin]

    # Backup old model
    if os.path.exists(model_path):
        backup_name = f"{coin}_model_{version}_backup.pkl"
        backup_path = os.path.join(BACKUP_DIR, backup_name)
        import shutil
        shutil.copy2(model_path, backup_path)
        print(f"  📦 Old model backed up → {backup_name}")

    # Save new model
    joblib.dump(new_model, model_path)
    print(f"  ✅ New model saved → {model_path}")


# ═══════════════════════════════════════════════════════════════
# BLOCK 8 — UPDATE retraining_log
# ═══════════════════════════════════════════════════════════════
def update_retraining_log(conn, coin: str, status: str,
                           accuracy_before: float,
                           accuracy_after: float,
                           train_start: date, train_end: date,
                           new_version: str, notes: str = ""):
    """Update all pending alerts for this coin."""
    cur = conn.cursor()
    cur.execute("""
        UPDATE retraining_log
        SET retrain_status          = %s,
            rolling_accuracy_after  = %s,
            training_data_start     = %s,
            training_data_end       = %s,
            new_model_version       = %s,
            notes                   = %s
        WHERE coin = %s
          AND retrain_status = 'pending'
    """, (
        status,
        accuracy_after,
        train_start,
        train_end,
        new_version,
        notes,
        coin,
    ))
    conn.commit()
    cur.close()
    print(f"  ✅ retraining_log updated → status: {status}")


# ═══════════════════════════════════════════════════════════════
# BLOCK 9 — DISPLAY RETRAINING LOG SUMMARY
# ═══════════════════════════════════════════════════════════════
def show_log_summary(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT coin, triggered_at, retrain_status,
               rolling_accuracy_before, rolling_accuracy_after,
               new_model_version, trigger_reason
        FROM retraining_log
        ORDER BY triggered_at DESC
        LIMIT 10
    """)
    rows = cur.fetchall()
    print(f"\n  {'Coin':<6} {'Triggered':<22} {'Status':<10} "
          f"{'Acc Before':>10} {'Acc After':>10} {'Version':>8}")
    print(f"  {'-'*72}")
    for row in rows:
        acc_before = f"{float(row[3])*100:.1f}%" if row[3] else "N/A"
        acc_after  = f"{float(row[4])*100:.1f}%" if row[4] else "N/A"
        version    = row[5] or "N/A"
        print(f"  {row[0]:<6} {str(row[1])[:21]:<22} {row[2]:<10} "
              f"{acc_before:>10} {acc_after:>10} {version:>8}")
    cur.close()


# ═══════════════════════════════════════════════════════════════
# MAIN RETRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════
def retrain_coin(conn, coin: str, accuracy_before: float):
    """Full retraining pipeline for one coin."""

    print(f"\n{'='*55}")
    print(f"  RETRAINING — {coin.upper()}")
    print(f"{'='*55}")

    new_version = f"v_{datetime.now().strftime('%Y%m%d_%H%M')}"

    try:
        # Step 1 — Fetch fresh data
        print("\n📥 Step 1 : Fetching fresh training data...")
        raw_df = fetch_training_data(coin)

        # Step 2 — Build features
        print("\n⚙️  Step 2 : Building features...")
        df = build_features(raw_df)

        # Step 3 — Evaluate old model first
        print("\n📊 Step 3 : Evaluating old model...")
        old_accuracy = evaluate_old_model(coin, df)

        # Step 4 — Train new model
        print("\n🔧 Step 4 : Training new model...")
        new_model, new_accuracy, train_start, train_end = train_and_evaluate(coin, df)

        # Step 5 — Compare
        print(f"\n📊 Step 5 : Comparison")
        print(f"  Old accuracy : {old_accuracy*100:.1f}%")
        print(f"  New accuracy : {new_accuracy*100:.1f}%")

        if new_accuracy >= old_accuracy:
            print(f"  ✅ New model is better — saving...")
            save_model(coin, new_model, new_version)
            status = "success"
            notes  = f"Improved from {old_accuracy*100:.1f}% to {new_accuracy*100:.1f}%"
        else:
            print(f"  ⚠️  New model is NOT better — keeping old model")
            status = "skipped"
            notes  = (f"New model ({new_accuracy*100:.1f}%) did not beat "
                      f"old model ({old_accuracy*100:.1f}%) — old model kept")

        # Step 6 — Update log
        print(f"\n📝 Step 6 : Updating retraining_log...")
        update_retraining_log(
            conn, coin, status,
            accuracy_before = accuracy_before,
            accuracy_after  = new_accuracy,
            train_start     = train_start,
            train_end       = train_end,
            new_version     = new_version,
            notes           = notes,
        )

        print(f"\n  ✅ Retraining complete for {coin.upper()} — {status.upper()}")

    except Exception as e:
        print(f"\n  ❌ Retraining failed for {coin.upper()}: {e}")
        update_retraining_log(
            conn, coin, "failed",
            accuracy_before = accuracy_before,
            accuracy_after  = None,
            train_start     = None,
            train_end       = None,
            new_version     = None,
            notes           = str(e),
        )


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", type=str, choices=["btc", "eth", "both"],
                        help="Force retrain a coin regardless of alert")
    args = parser.parse_args()

    print("\n" + "="*55)
    print(f"  STEP 6 — Retraining Script")
    print(f"  Date : {date.today()}")
    print("="*55)

    conn = psycopg2.connect(**DB_CONFIG)
    print("  ✅ Connected to crypto_tracker")

    # ── Forced retraining
    if args.force:
        coins = ["btc", "eth"] if args.force == "both" else [args.force]
        for coin in coins:
            print(f"\n  ⚡ Force retraining {coin.upper()}...")
            retrain_coin(conn, coin, accuracy_before=0.0)

    else:
        # ── Check for pending alerts
        print("\n🔍 Checking for pending retraining alerts...")
        alerts = get_pending_alerts(conn)

        if not alerts:
            print("  ✅ No pending alerts — all models are healthy!")
        else:
            print(f"  🚨 Found {len(alerts)} pending alert(s)")
            seen_coins = set()
            for alert in alerts:
                coin, triggered_at, reason, acc_before = alert
                if coin in seen_coins:
                    continue
                seen_coins.add(coin)
                print(f"\n  Alert : {coin.upper()}")
                print(f"  Reason: {reason}")
                retrain_coin(conn, coin, float(acc_before) if acc_before else 0.0)

    # Summary
    print(f"\n{'─'*55}")
    print("  📊 retraining_log summary:")
    show_log_summary(conn)

    conn.close()

    print("\n" + "="*55)
    print("  ✅ Step 6 complete!")
    print("  Next: set up Task Scheduler (Step 7)")
    print("="*55)