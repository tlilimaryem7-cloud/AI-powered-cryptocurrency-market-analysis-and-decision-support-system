"""
Cryptocurrency Data Pipeline
============================
Preprocessing and feature engineering for crypto price prediction.

Author: [Maryem Tlili]
Date: 2026-02-11
"""

import pandas as pd
import numpy as np


def preprocess_and_engineer_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Complete preprocessing and feature engineering pipeline for crypto data.
    
    This function takes raw data from CoinGecko API and transforms it into
    a clean dataset ready for machine learning modeling.
    
    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw dataframe with columns: timestamp, price, market_cap, volume, coin,
        market_cap_rank, circulating_supply, max_supply, ath, atl
    
    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with engineered features, sorted by coin and timestamp,
        with all NaNs filled and partial-day data removed.
    
    Features Created
    ----------------
    - log_return_1d: Daily log returns
    - volatility_7d, volatility_14d: Rolling volatility
    - rsi_14: Relative Strength Index
    - volume_to_marketcap: Volume ratio
    - circulating_supply_history: Historical supply calculation
    - distance_from_ath, distance_from_atl: Distance from extremes
    - has_max_supply: Binary flag for capped supply
    - ma_7, ma_14, ma_30: Moving averages
    - price_to_ma7, price_to_ma30: Price relative to MAs
    - macd, macd_signal, macd_histogram: MACD indicators
    - momentum_5d, momentum_10d, momentum_acceleration: Momentum indicators
    
    Example
    -------
    >>> raw_data = fetch_coin_history("bitcoin", days=365)
    >>> clean_data = preprocess_and_engineer_features(raw_data)
    >>> print(clean_data.shape)
    (365, 26)
    """
    
    df = df_raw.copy()
    
    print("Starting preprocessing pipeline...")
    

    # SECTION 1: PREPROCESSING
    
    print("  ├─ Rounding values...")
    df["price"] = df["price"].round(2)
    df["market_cap"] = df["market_cap"].round(0)
    df["volume"] = df["volume"].round(0)
    df["circulating_supply"] = df["circulating_supply"].round(0)
    df["max_supply"] = df["max_supply"].round(0)
    df["ath"] = df["ath"].round(2)
    df["atl"] = df["atl"].round(2)
    
    print("  ├─ Converting timestamp...")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    print("  ├─ Sorting by coin and timestamp...")
    df = df.sort_values(["coin", "timestamp"]).reset_index(drop=True)
    
    print("  ├─ Removing partial-day data...")
    # Compute time differences to find incomplete days
    df["time_diff"] = df.groupby("coin")["timestamp"].diff()
    
    # Identify bad rows (not exactly 1 day apart, excluding first row per coin)
    bad_rows = df[
        df["time_diff"].notna() &
        (df["time_diff"] != pd.Timedelta(days=1))
    ]
    
    if len(bad_rows) > 0:
        print(f"    └─ Dropped {len(bad_rows)} partial-day rows")
        df = df.drop(bad_rows.index).reset_index(drop=True)
    
    # Drop the temporary time_diff column
    df = df.drop(columns=["time_diff"])
    
    # SECTION 2: FEATURE ENGINEERING

    print("  ├─ Engineering features...")
    
    # --- Log Returns ---
    print("    ├─ Log returns...")
    df["log_return_1d"] = (
        df.groupby("coin")["price"]
        .transform(lambda x: np.log(x / x.shift(1)))
    )
    
    # --- Volatility ---
    print("    ├─ Volatility indicators...")
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
    
    # --- RSI ---
    print("    ├─ RSI...")
    window = 14
    delta = df.groupby("coin")["price"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.groupby(df["coin"]).rolling(window).mean().reset_index(level=0, drop=True)
    avg_loss = loss.groupby(df["coin"]).rolling(window).mean().reset_index(level=0, drop=True)
    
    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))
    
    # --- Volume Ratio ---
    print("    ├─ Volume ratio...")
    df["volume_to_marketcap"] = df["volume"] / df["market_cap"]
    
    # --- Circulating Supply History ---
    print("    ├─ Circulating supply history...")
    df["circulating_supply_history"] = (df["market_cap"] / df["price"]).round(0)
    
    # --- Distance from ATH/ATL ---
    print("    ├─ Distance from extremes...")
    df["distance_from_ath"] = (df["price"] - df["ath"]) / df["ath"]
    df["distance_from_atl"] = (df["price"] - df["atl"]) / df["atl"]
    
    # --- Max Supply Flag ---
    print("    ├─ Max supply flag...")
    df["has_max_supply"] = df["max_supply"].notna().astype(int)
    
    # --- Moving Averages ---
    print("    ├─ Moving averages...")
    df["ma_7"] = df.groupby("coin")["price"].rolling(7).mean().reset_index(level=0, drop=True)
    df["ma_14"] = df.groupby("coin")["price"].rolling(14).mean().reset_index(level=0, drop=True)
    df["ma_30"] = df.groupby("coin")["price"].rolling(30).mean().reset_index(level=0, drop=True)
   
    # Price relative to MAs
    df["price_to_ma7"] = (df["price"] - df["ma_7"]) / df["ma_7"]
    df["price_to_ma30"] = (df["price"] - df["ma_30"]) / df["ma_30"]
    
    df["price_to_ma7"] = df["price_to_ma7"].fillna(0)
    df["price_to_ma30"] = df["price_to_ma30"].fillna(0)
    
    # --- MACD ---
    print("    ├─ MACD indicators...")
    ema_12 = df.groupby("coin")["price"].ewm(span=12, adjust=False).mean().reset_index(level=0, drop=True)
    ema_26 = df.groupby("coin")["price"].ewm(span=26, adjust=False).mean().reset_index(level=0, drop=True)
    
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df.groupby("coin")["macd"].ewm(span=9, adjust=False).mean().reset_index(level=0, drop=True)
    df["macd_histogram"] = df["macd"] - df["macd_signal"]
    
    # --- Momentum ---
    print("    ├─ Momentum indicators...")
    df["momentum_5d"] = df.groupby("coin")["price"].pct_change(periods=5)
    df["momentum_10d"] = df.groupby("coin")["price"].pct_change(periods=10)
    df["momentum_acceleration"] = df["momentum_5d"] - df["momentum_10d"]

    # SECTION 3: FILL NaNs
    
    print("  ├─ Filling NaNs...")
    
    # Volatility: Fill with 0 (no volatility yet)
    df["volatility_7d"] = df["volatility_7d"].fillna(0)
    df["volatility_14d"] = df["volatility_14d"].fillna(0)

    # Fill MA NaNs with current price
    df["ma_7"] = df["ma_7"].fillna(df["price"])
    df["ma_14"] = df["ma_14"].fillna(df["price"])
    df["ma_30"] = df["ma_30"].fillna(df["price"])
    
    # RSI: Fill with 50 (neutral)
    df["rsi_14"] = df["rsi_14"].fillna(50)
    
    # MACD: Fill with 0 (no signal yet)
    df["macd"] = df["macd"].fillna(0)
    df["macd_signal"] = df["macd_signal"].fillna(0)
    df["macd_histogram"] = df["macd_histogram"].fillna(0)
    
    # Momentum: Fill with 0 (no momentum yet)
    df["momentum_5d"] = df["momentum_5d"].fillna(0)
    df["momentum_10d"] = df["momentum_10d"].fillna(0)
    df["momentum_acceleration"] = df["momentum_acceleration"].fillna(0)
   
    # SECTION 4: CLEANUP
    
    print("  ├─ Dropping unused columns...")
    df = df.drop(columns=["circulating_supply", "max_supply", "total_supply"], errors="ignore")
 
    # SECTION 5: DEFINE FINAL COLUMN ORDER
    
    print("  ├─ Enforcing column order...")
    FINAL_COLUMNS = [
        # Identifiers
        "timestamp", "coin",
        
        # Raw features
        "price", "market_cap", "volume", "market_cap_rank",
        "ath", "atl",
        
        # Engineered features
        "log_return_1d", "volatility_7d", "volatility_14d", "rsi_14",
        "volume_to_marketcap", "circulating_supply_history",
        "distance_from_ath", "distance_from_atl", "has_max_supply",
        
        # Moving averages
        "ma_7", "ma_14", "ma_30", "price_to_ma7", "price_to_ma30",
        
        # MACD
        "macd", "macd_signal", "macd_histogram",
        
        # Momentum
        "momentum_5d", "momentum_10d", "momentum_acceleration"
    ]
    
    df = df[FINAL_COLUMNS]
    
    # SECTION 6: VERIFICATION
    
    print("  └─ Running verification checks...")
    
    print(f"\n Pipeline Complete!")
    print(f"   Total rows: {len(df)}")
    print(f"   Total columns: {len(df.columns)}")
    print(f"   Coins: {df['coin'].value_counts().to_dict()}")
    print(f"   Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    
    # Check for NaNs
    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        print(f"\n⚠️  Warning: Found NaNs in columns:")
        print(nan_counts[nan_counts > 0])
    else:
        print(f"   No NaNs found ✓")
    
    # Check for duplicates
    duplicates = df.duplicated(subset=["timestamp", "coin"]).sum()
    if duplicates > 0:
        print(f"\n⚠️  Warning: Found {duplicates} duplicate timestamp-coin pairs")
    else:
        print(f"   No duplicates found ✓")
    
    return df


# HELPER FUNCTION: Save Cleaned Data

def save_cleaned_data(df: pd.DataFrame, output_path: str = "../data/processed"):
    """
    Save cleaned dataframe to CSV with timestamp in filename.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe from preprocess_and_engineer_features()
    output_path : str
        Directory to save the file (default: ../data/processed)
    
    Returns
    -------
    str
        Path to saved file
    """
    import os
    
    os.makedirs(output_path, exist_ok=True)
    
    # Generate filename with current date
    snapshot_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    file_path = f"{output_path}/crypto_clean_{snapshot_date}.csv"
    
    # Save
    df.to_csv(file_path, index=False)
    
    print(f"💾 Saved cleaned data to: {file_path}")
    
    return file_path