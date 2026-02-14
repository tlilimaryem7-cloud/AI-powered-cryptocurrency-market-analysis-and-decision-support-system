# ============================================================
# PIPELINE — BTC/ETH CRYPTO FEATURE ENGINEERING
# ============================================================
# Input  : fetches live data from yfinance + alternative.me
# Output : data/processed/crypto_features.csv
#
# Run    : python pipeline.py
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os

# ─────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────
COINS         = ["BTC-USD", "ETH-USD"]
START_DATE    = "2017-01-01"
END_DATE      = "2026-02-12"
RAW_DATA_PATH       = "data/raw"
PROCESSED_DATA_PATH = "data/processed"
os.makedirs(RAW_DATA_PATH,       exist_ok=True)
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
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
    df = df.drop(columns=[close_col])
    return df


def fetch_fear_greed(limit=3000):
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


# ═══════════════════════════════════════════════════════════════
# BLOCK 1 — CRYPTO PRICES
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("📥 BLOCK 1 — CRYPTO PRICES")
print("="*60)

all_crypto = []
for ticker in COINS:
    print(f"\n  Downloading {ticker}...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE,
                     interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        print(f"  ⚠️  {ticker} empty — skipping")
        continue
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df = df.rename(columns={"Date":"timestamp", "Close":"price", "Volume":"volume"})
    df = df[["timestamp", "price", "volume"]]
    df["coin"] = ticker.split("-")[0].lower()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"  ✅ {ticker}: {len(df)} rows")
    all_crypto.append(df)

crypto_df = pd.concat(all_crypto, ignore_index=True)
crypto_df = crypto_df.sort_values(["coin", "timestamp"]).reset_index(drop=True)
print(f"\n  Combined crypto shape: {crypto_df.shape}")


# ═══════════════════════════════════════════════════════════════
# BLOCK 2 — MACRO SIGNALS
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("📥 BLOCK 2 — MACRO SIGNALS (yfinance)")
print("="*60)

spy_df = download_yf("SPY",      START_DATE, END_DATE, "spy_close")
dxy_df = download_yf("DX-Y.NYB", START_DATE, END_DATE, "dxy_close")
vix_df = download_yf("^VIX",     START_DATE, END_DATE, "vix")


# ═══════════════════════════════════════════════════════════════
# BLOCK 3 — FEAR & GREED INDEX
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("📥 BLOCK 3 — FEAR & GREED INDEX (alternative.me)")
print("="*60)

fg_df = fetch_fear_greed(limit=3000)


# ═══════════════════════════════════════════════════════════════
# BLOCK 4 — MACRO RETURNS
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("📥 BLOCK 4 — MACRO RETURNS")
print("="*60)

if spy_df is not None:
    spy_df = add_log_return(spy_df, "spy_close", "spy_return")
    print(f"  SPY return — mean: {spy_df['spy_return'].mean():.5f} | "
          f"std: {spy_df['spy_return'].std():.5f}")

if dxy_df is not None:
    dxy_df = add_log_return(dxy_df, "dxy_close", "dxy_return")
    print(f"  DXY return — mean: {dxy_df['dxy_return'].mean():.5f} | "
          f"std: {dxy_df['dxy_return'].std():.5f}")

if vix_df is not None:
    vix_df["vix_change"] = vix_df["vix"].diff()
    print(f"  VIX — mean: {vix_df['vix'].mean():.2f} | "
          f"std: {vix_df['vix'].std():.2f}")


# ═══════════════════════════════════════════════════════════════
# BLOCK 5 — MERGE
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("📥 BLOCK 5 — MERGING ALL SIGNALS")
print("="*60)

df = crypto_df.copy()

for name, ext_df in [
    ("SPY",        spy_df),
    ("DXY",        dxy_df),
    ("VIX",        vix_df),
    ("Fear&Greed", fg_df),
]:
    if ext_df is None:
        print(f"  ⚠️  {name} skipped (download failed)")
        continue
    before = df.shape[1]
    df = df.merge(ext_df, on="timestamp", how="left")
    print(f"  ✅ {name} merged — +{df.shape[1] - before} col(s)")

# Save raw
raw_path = os.path.join(RAW_DATA_PATH, "crypto_raw.csv")
df.to_csv(raw_path, index=False)
print(f"\n✅ Raw saved: {raw_path}")


# ═══════════════════════════════════════════════════════════════
# BLOCK 6 — PREPROCESSING
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("⚙️  BLOCK 6 — PREPROCESSING")
print("="*60)

# ── Sort
df = df.sort_values(["coin", "timestamp"]).reset_index(drop=True)

# ── Rounding
df["price"]      = df["price"].round(2)
df["volume"]     = df["volume"].round(0).astype("Int64")
df["spy_return"] = df["spy_return"].round(6)
df["dxy_return"] = df["dxy_return"].round(6)
df["vix"]        = df["vix"].round(2)
df["vix_change"] = df["vix_change"].round(2)
df["fear_greed"] = df["fear_greed"].round(0).astype("Int64")

# ── Forward-fill macro (weekends/holidays — max 3 days)
macro_cols = ["spy_return", "dxy_return", "vix", "vix_change"]
df[macro_cols] = (
    df.groupby("coin")[macro_cols]
      .transform(lambda x: x.ffill(limit=3))
)

# ── Forward-fill fear & greed
df["fear_greed"] = (
    df.groupby("coin")["fear_greed"]
      .transform(lambda x: x.ffill())
)

# ── Cap vix_change outliers (1st/99th percentile)
p01 = df["vix_change"].quantile(0.01)
p99 = df["vix_change"].quantile(0.99)
df["vix_change"] = df["vix_change"].clip(lower=p01, upper=p99)
print(f"  vix_change capped to [{p01:.2f}, {p99:.2f}]")

# ── Fear & Greed category
bins   = [0, 24, 44, 55, 74, 100]
labels = [0, 1, 2, 3, 4]
df["fear_greed_cat"] = pd.cut(
    df["fear_greed"],
    bins=bins,
    labels=labels,
    include_lowest=True
).astype("Int64")

print("  Preprocessing done ✅")


# ═══════════════════════════════════════════════════════════════
# BLOCK 7 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("⚙️  BLOCK 7 — FEATURE ENGINEERING")
print("="*60)

# ── Log return
print("  ├─ Log return...")
df["log_return_1d"] = (
    df.groupby("coin")["price"]
      .transform(lambda x: np.log(x / x.shift(1)))
)

# ── Volatility
print("  ├─ Volatility...")
df["volatility_7d"] = (
    df.groupby("coin")["log_return_1d"]
      .transform(lambda x: x.rolling(7).std())
)
df["volatility_14d"] = (
    df.groupby("coin")["log_return_1d"]
      .transform(lambda x: x.rolling(14).std())
)
df["volatility_21d"] = (
    df.groupby("coin")["log_return_1d"]
      .transform(lambda x: x.rolling(21).std())
)

# ── RSI
print("  ├─ RSI...")
delta    = df.groupby("coin")["price"].transform(lambda x: x.diff())
gain     = delta.clip(lower=0)
loss     = -delta.clip(upper=0)
avg_gain = gain.groupby(df["coin"]).transform(lambda x: x.rolling(14).mean())
avg_loss = loss.groupby(df["coin"]).transform(lambda x: x.rolling(14).mean())
df["rsi_14"] = 100 - (100 / (1 + avg_gain / avg_loss))

# ── Moving Averages
print("  ├─ Moving averages...")
for w in [7, 14, 30, 50]:
    df[f"ma_{w}"] = (
        df.groupby("coin")["price"]
          .transform(lambda x: x.rolling(w).mean())
    )

df["price_to_ma7"]  = (df["price"] - df["ma_7"])  / df["ma_7"]
df["price_to_ma30"] = (df["price"] - df["ma_30"]) / df["ma_30"]
df["price_to_ma50"] = (df["price"] - df["ma_50"]) / df["ma_50"]

# ── MACD
print("  ├─ MACD...")
ema_12 = (
    df.groupby("coin")["price"]
      .transform(lambda x: x.ewm(span=12, adjust=False).mean())
)
ema_26 = (
    df.groupby("coin")["price"]
      .transform(lambda x: x.ewm(span=26, adjust=False).mean())
)
df["macd"]           = ema_12 - ema_26
df["macd_signal"]    = (
    df.groupby("coin")["macd"]
      .transform(lambda x: x.ewm(span=9, adjust=False).mean())
)
df["macd_histogram"] = df["macd"] - df["macd_signal"]

# ── Momentum
print("  ├─ Momentum...")
df["momentum_5d"]  = (
    df.groupby("coin")["price"]
      .transform(lambda x: x.pct_change(5))
)
df["momentum_10d"] = (
    df.groupby("coin")["price"]
      .transform(lambda x: x.pct_change(10))
)
df["momentum_acceleration"] = df["momentum_5d"] - df["momentum_10d"]

# ── Bollinger Bands
print("  ├─ Bollinger Bands...")
bb_std         = (
    df.groupby("coin")["price"]
      .transform(lambda x: x.rolling(14).std())
)
df["bb_width"] = (4 * bb_std) / df["ma_14"]
df["bb_pct"]   = (
    (df["price"] - (df["ma_14"] - 2 * bb_std)) /
    (4 * bb_std)
)

# ── Macro rolling features
print("  ├─ Macro rolling features...")
df["spy_return_ma7"]  = (
    df.groupby("coin")["spy_return"]
      .transform(lambda x: x.rolling(7).mean())
)
df["spy_return_std7"] = (
    df.groupby("coin")["spy_return"]
      .transform(lambda x: x.rolling(7).std())
)
df["dxy_return_ma7"]  = (
    df.groupby("coin")["dxy_return"]
      .transform(lambda x: x.rolling(7).mean())
)
df["vix_ma14"]        = (
    df.groupby("coin")["vix"]
      .transform(lambda x: x.rolling(14).mean())
)
df["vix_regime"]      = (df["vix"] > 20).astype(int)

# ── Sentiment features
print("  ├─ Sentiment features...")
df["fear_greed_ma7"]  = (
    df.groupby("coin")["fear_greed"]
      .transform(lambda x: x.rolling(7).mean())
)
df["fear_greed_lag1"] = (
    df.groupby("coin")["fear_greed"]
      .transform(lambda x: x.shift(1))
)
df["fear_greed_lag7"] = (
    df.groupby("coin")["fear_greed"]
      .transform(lambda x: x.shift(7))
)

# ── Regime flags
print("  ├─ Regime flags...")
df["bull_bear_flag"]  = (df["price"] > df["ma_50"]).astype(int)
vol_median            = (
    df.groupby("coin")["volatility_21d"]
      .transform(lambda x: x.rolling(90).median())
)
df["volatility_regime"] = (df["volatility_21d"] > vol_median).astype(int)

# ── Log transforms (after all rolling features)
print("  ├─ Log transforms...")
df["log_price"]  = np.log(df["price"])
df["log_volume"] = np.log(df["volume"].astype(float))

print("  └─ Feature engineering complete ✅")


# ═══════════════════════════════════════════════════════════════
# BLOCK 8 — DROP NaNs
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("🧹 BLOCK 8 — DROP NaNs")
print("="*60)

before = len(df)
df     = df.dropna().reset_index(drop=True)
after  = len(df)

print(f"  Rows before  : {before}")
print(f"  Rows dropped : {before - after}")
print(f"  Rows kept    : {after}")
print(f"  NaN check    : {df.isna().sum().sum()} (should be 0)")


# ═══════════════════════════════════════════════════════════════
# BLOCK 9 — SAVE
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("💾 BLOCK 9 — SAVE")
print("="*60)

processed_path = os.path.join(PROCESSED_DATA_PATH, "crypto_features.csv")
df.to_csv(processed_path, index=False)

print(f"  Shape      : {df.shape}")
print(f"  Coins      : {df['coin'].value_counts().to_dict()}")
print(f"  Date range : {df['timestamp'].min().date()} → "
      f"{df['timestamp'].max().date()}")
print(f"\n  Columns:")
for i, col in enumerate(df.columns):
    print(f"    {i:02d}. {col}")

print(f"\n✅ Saved: {processed_path}")