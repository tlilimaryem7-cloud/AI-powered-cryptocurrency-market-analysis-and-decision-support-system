# ============================================================
# STEP 2 — Daily Data Fetch → raw_prices table
# ============================================================
# Fetches today's BTC and ETH price + macro signals
# and inserts one row per coin into the raw_prices table.
#
# Run daily (e.g. at midnight after market close)
# Run: python daily_fetch.py
# ============================================================

import sys
import os
import psycopg2
import yfinance as yf
import requests
import pandas as pd
import numpy as np
from datetime import date, datetime
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ─────────────────────────────────────────────────────────────
# PATH SETUP — point to your project root
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
# SETTINGS
# ─────────────────────────────────────────────────────────────
COINS = ["btc", "eth"]
TICKERS = {
    "btc": "BTC-USD",
    "eth": "ETH-USD",
}
TODAY = date.today()


# ─────────────────────────────────────────────────────────────
# BLOCK 1 — FETCH CRYPTO PRICE + VOLUME
# ─────────────────────────────────────────────────────────────
def fetch_crypto(coin: str) -> dict:
    """Fetch latest closing price and volume for a coin via yfinance."""
    ticker = TICKERS[coin]
    df = yf.download(ticker, period="2d", interval="1d",
                     auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"❌ yfinance returned empty data for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    latest = df.iloc[-1]

    return {
        "price" : round(float(latest["Close"]), 2),
        "volume": int(latest["Volume"]),
    }


# ─────────────────────────────────────────────────────────────
# BLOCK 2 — FETCH MACRO SIGNALS
# ─────────────────────────────────────────────────────────────
def fetch_macro() -> dict:
    """Fetch SPY return, DXY return, and VIX from yfinance."""
    result = {}

    for name, ticker, col in [
        ("spy_return", "SPY",      "spy"),
        ("dxy_return", "DX-Y.NYB", "dxy"),
        ("vix",        "^VIX",     "vix"),
    ]:
        df = yf.download(ticker, period="5d", interval="1d",
                         auto_adjust=True, progress=False)
        if df.empty:
            print(f"  ⚠️  {ticker} returned empty — setting None")
            result[name] = None
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index().sort_values("Date")
        closes = df["Close"].values

        if name == "vix":
            result["vix"] = round(float(closes[-1]), 2)
        else:
            if len(closes) >= 2:
                log_ret = float(np.log(closes[-1] / closes[-2]))
                result[name] = round(log_ret, 6)
            else:
                result[name] = None

    return result


# ─────────────────────────────────────────────────────────────
# BLOCK 3 — FETCH FEAR & GREED INDEX
# ─────────────────────────────────────────────────────────────
def fetch_fear_greed() -> int | None:
    """Fetch today's Fear & Greed index from alternative.me."""
    try:
        url  = "https://api.alternative.me/fng/?limit=1&format=json"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        val  = int(resp.json()["data"][0]["value"])
        return val
    except Exception as e:
        print(f"  ⚠️  Fear & Greed fetch failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# BLOCK 4 — INSERT INTO raw_prices
# ─────────────────────────────────────────────────────────────
def insert_raw_price(conn, coin: str, crypto: dict,
                     macro: dict, fear_greed: int | None):
    """Insert one row into raw_prices. Skip if already exists for today."""
    cur = conn.cursor()

    # Check if already inserted today
    cur.execute(
        "SELECT id FROM raw_prices WHERE coin = %s AND fetch_date = %s",
        (coin, TODAY)
    )
    if cur.fetchone():
        print(f"  ℹ️  {coin.upper()} — already in raw_prices for {TODAY}, skipping")
        cur.close()
        return False

    cur.execute("""
        INSERT INTO raw_prices
            (coin, fetch_date, price, volume, fear_greed,
             vix, spy_return, dxy_return)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        coin,
        TODAY,
        crypto["price"],
        crypto["volume"],
        fear_greed,
        macro.get("vix"),
        macro.get("spy_return"),
        macro.get("dxy_return"),
    ))

    conn.commit()
    cur.close()
    return True


# ─────────────────────────────────────────────────────────────
# BLOCK 5 — DISPLAY SUMMARY
# ─────────────────────────────────────────────────────────────
def show_summary(conn):
    """Print current state of raw_prices table."""
    cur = conn.cursor()
    cur.execute("""
        SELECT coin, fetch_date, price, volume, fear_greed, vix
        FROM raw_prices
        ORDER BY fetch_date DESC, coin
        LIMIT 10
    """)
    rows = cur.fetchall()
    print(f"\n  {'Coin':<6} {'Date':<12} {'Price':>12} {'Volume':>14} {'F&G':>5} {'VIX':>6}")
    print(f"  {'-'*58}")
    for row in rows:
        print(f"  {row[0]:<6} {str(row[1]):<12} {float(row[2]):>12,.2f} "
              f"{int(row[3]) if row[3] else 0:>14,} "
              f"{int(row[4]) if row[4] else 0:>5} "
              f"{float(row[5]) if row[5] else 0:>6.2f}")
    cur.close()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print(f"  STEP 2 — Daily Fetch -> raw_prices")
    print(f"  Date : {TODAY}")
    print("="*55)

    # Connect to DB
    conn = psycopg2.connect(**DB_CONFIG)
    print("  ✅ Connected to crypto_tracker")

    # Fetch macro once (shared across coins)
    print("\n📥 Fetching macro signals...")
    macro = fetch_macro()
    print(f"  SPY return : {macro.get('spy_return')}")
    print(f"  DXY return : {macro.get('dxy_return')}")
    print(f"  VIX        : {macro.get('vix')}")

    # Fetch Fear & Greed once
    print("\n📥 Fetching Fear & Greed...")
    fear_greed = fetch_fear_greed()
    print(f"  Fear & Greed : {fear_greed}")

    # Fetch + insert each coin
    for coin in COINS:
        print(f"\n📥 Fetching {coin.upper()}...")
        try:
            crypto  = fetch_crypto(coin)
            print(f"  Price  : ${crypto['price']:,.2f}")
            print(f"  Volume : {crypto['volume']:,}")

            inserted = insert_raw_price(conn, coin, crypto, macro, fear_greed)
            if inserted:
                print(f"  ✅ {coin.upper()} inserted into raw_prices")

        except Exception as e:
            print(f"  ❌ {coin.upper()} failed: {e}")

    # Summary
    print("\n📊 raw_prices table (last 10 rows):")
    show_summary(conn)

    conn.close()

    print("\n" + "="*55)
    print("  ✅ Step 2 complete!")
    print("  Next: run predict_and_store.py (Step 3)")
    print("="*55)