# ============================================
# CRYPTO PREPROCESSING & FEATURE ENGINEERING
# PIPELINE v2 - yfinance compatible
# ============================================

import pandas as pd
import numpy as np
from datetime import datetime


def preprocess_and_engineer_features(df_raw):
    """
    Preprocess raw yfinance crypto data and engineer features.
    
    Input columns:  timestamp, coin, price, volume
    Output columns: timestamp, coin, price, volume +
                    all engineered features
    """
    
    print("="*60)
    print("CRYPTO FEATURE ENGINEERING PIPELINE v2")
    print("="*60)
    
    df = df_raw.copy()
    
    # ============================================
    # STEP 1: PREPROCESSING
    # ============================================
    print("\n📋 STEP 1: PREPROCESSING")
    
    # ── Round values ──
    print("  ├─ Rounding numerical values...")
    df["price"]  = df["price"].round(2)
    df["volume"] = df["volume"].round(0)
    
    # ── Sort ──
    print("  ├─ Sorting by coin and timestamp...")
    df = df.sort_values(
        ["coin", "timestamp"]
    ).reset_index(drop=True)
    
    print(f"  └─ Preprocessing complete! Shape: {df.shape}")
    
    # ============================================
    # STEP 2: FEATURE ENGINEERING
    # ============================================
    print("\n📋 STEP 2: FEATURE ENGINEERING")
    
    # ── Log Returns ──
    print("  ├─ Calculating log returns...")
    df["log_return_1d"] = (
        df.groupby("coin")["price"]
        .transform(lambda x: np.log(x / x.shift(1)))
    )
    
    # ── Volatility ──
    print("  ├─ Calculating volatility...")
    df["volatility_7d"] = (
        df.groupby("coin")["log_return_1d"]
        .rolling(window=7)
        .std()
        .reset_index(level=0, drop=True)
    )
    df["volatility_14d"] = (
        df.groupby("coin")["log_return_1d"]
        .rolling(window=14)
        .std()
        .reset_index(level=0, drop=True)
    )
    
    # ── RSI ──
    print("  ├─ Calculating RSI...")
    window   = 14
    delta    = df.groupby("coin")["price"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = (
        gain.groupby(df["coin"])
        .rolling(window)
        .mean()
        .reset_index(level=0, drop=True)
    )
    avg_loss = (
        loss.groupby(df["coin"])
        .rolling(window)
        .mean()
        .reset_index(level=0, drop=True)
    )
    rs           = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))
    
    # ── Fill NaNs: Volatility & RSI ──
    print("  ├─ Filling NaNs (volatility & RSI)...")
    df["volatility_7d"]  = df["volatility_7d"].fillna(0)
    df["volatility_14d"] = df["volatility_14d"].fillna(0)
    df["rsi_14"]         = df["rsi_14"].fillna(50)
    
    # ── Moving Averages ──
    print("  ├─ Calculating moving averages...")
    for w in [7, 14, 30]:
        df[f"ma_{w}"] = (
            df.groupby("coin")["price"]
            .rolling(w)
            .mean()
            .reset_index(level=0, drop=True)
        )
    
    df["price_to_ma7"]  = (df["price"] - df["ma_7"])  / df["ma_7"]
    df["price_to_ma30"] = (df["price"] - df["ma_30"]) / df["ma_30"]
    
    # ── Fill NaNs: MAs ──
    print("  ├─ Filling NaNs (moving averages)...")
    for w in [7, 14, 30]:
        df[f"ma_{w}"] = df[f"ma_{w}"].fillna(df["price"])
    df["price_to_ma7"]  = df["price_to_ma7"].fillna(0)
    df["price_to_ma30"] = df["price_to_ma30"].fillna(0)
    
    # ── MACD ──
    print("  ├─ Calculating MACD...")
    ema_12 = (
        df.groupby("coin")["price"]
        .ewm(span=12, adjust=False)
        .mean()
        .reset_index(level=0, drop=True)
    )
    ema_26 = (
        df.groupby("coin")["price"]
        .ewm(span=26, adjust=False)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = (
        df.groupby("coin")["macd"]
        .ewm(span=9, adjust=False)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["macd_histogram"] = df["macd"] - df["macd_signal"]
    
    # ── Fill NaNs: MACD ──
    print("  ├─ Filling NaNs (MACD)...")
    df["macd"]           = df["macd"].fillna(0)
    df["macd_signal"]    = df["macd_signal"].fillna(0)
    df["macd_histogram"] = df["macd_histogram"].fillna(0)
    
    # ── Momentum ──
    print("  ├─ Calculating momentum...")
    df["momentum_5d"]  = (
        df.groupby("coin")["price"]
        .pct_change(periods=5)
    )
    df["momentum_10d"] = (
        df.groupby("coin")["price"]
        .pct_change(periods=10)
    )
    df["momentum_acceleration"] = (
        df["momentum_5d"] - df["momentum_10d"]
    )
    
    # ── Fill NaNs: Momentum ──
    print("  ├─ Filling NaNs (momentum)...")
    df["momentum_5d"]           = df["momentum_5d"].fillna(0)
    df["momentum_10d"]          = df["momentum_10d"].fillna(0)
    df["momentum_acceleration"] = df["momentum_acceleration"].fillna(0)
    
    # ── Fill NaNs: Log Return ──
    df["log_return_1d"] = df["log_return_1d"].fillna(0)
    
    print("  └─ Feature engineering complete!")
    
    # ============================================
    # STEP 3: COLUMN ORDER
    # ============================================
    print("\n📋 STEP 3: ORDERING COLUMNS")
    
    final_cols = [
        "timestamp", "coin", "price", "volume",
        "log_return_1d",
        "volatility_7d", "volatility_14d",
        "rsi_14",
        "ma_7", "ma_14", "ma_30",
        "price_to_ma7", "price_to_ma30",
        "macd", "macd_signal", "macd_histogram",
        "momentum_5d", "momentum_10d",
        "momentum_acceleration"
    ]
    
    df = df[final_cols]
    print(f"  └─ {len(final_cols)} columns ordered!")
    
    # ============================================
    # STEP 4: VERIFICATION
    # ============================================
    print("\n📋 STEP 4: VERIFICATION")
    print(f"  ├─ Shape:      {df.shape}")
    print(f"  ├─ NaNs:       {df.isna().sum().sum()}")
    print(f"  ├─ Duplicates: {df.duplicated().sum()}")
    print(f"  ├─ Coins:      {df['coin'].value_counts().to_dict()}")
    print(f"  └─ Date range: {df['timestamp'].min().date()} → "
          f"{df['timestamp'].max().date()}")
    
    return df


def save_cleaned_data(df, output_path):
    """Save cleaned dataframe with date in filename."""
    
    today     = datetime.today().strftime("%Y-%m-%d")
    filename  = f"crypto_clean_{today}.csv"
    full_path = f"{output_path}/{filename}"
    
    df.to_csv(full_path, index=False)
    
    print(f"\n✅ Saved: {full_path}")
    print(f"   Rows:    {len(df)}")
    print(f"   Columns: {len(df.columns)}")
    
    return full_path