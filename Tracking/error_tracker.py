# ============================================================
# STEP 4 — Error Calculation → errors table
# ============================================================
# Runs daily AFTER daily_fetch.py
# Looks at yesterday's predictions, compares to true price
# movement now available in raw_prices, computes accuracy,
# and inserts results into the errors table.
#
# Also computes rolling 30-day accuracy and flags if
# retraining is needed.
#
# Run: python error_tracker.py
# For simulation mode: python error_tracker.py --simulate
# ============================================================

import sys
import os
import psycopg2
import pandas as pd
import argparse
from datetime import date, timedelta

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
COINS              = ["btc", "eth"]
ACCURACY_THRESHOLD = 0.52   # below this → flag needs_retraining = True
ROLLING_WINDOW     = 30     # days for rolling accuracy


# ─────────────────────────────────────────────────────────────
# BLOCK 1 — FETCH PENDING PREDICTIONS
# Predictions where target_date is in raw_prices but not yet
# in errors table
# ─────────────────────────────────────────────────────────────
def get_pending_predictions(conn, coin: str) -> list:
    """
    Find predictions whose target_date now has a true price
    in raw_prices, but haven't been evaluated yet in errors.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT
            p.id,
            p.coin,
            p.prediction_date,
            p.target_date,
            p.predicted_direction,
            p.confidence
        FROM predictions p
        -- true price must exist for the target date
        INNER JOIN raw_prices r
            ON r.coin = p.coin
            AND r.fetch_date = p.target_date
        -- not already evaluated
        LEFT JOIN errors e
            ON e.coin = p.coin
            AND e.prediction_date = p.prediction_date
        WHERE p.coin = %s
          AND e.id IS NULL
        ORDER BY p.prediction_date ASC
    """, (coin,))
    rows = cur.fetchall()
    cur.close()
    return rows


# ─────────────────────────────────────────────────────────────
# BLOCK 2 — GET TRUE DIRECTION
# Compare price on target_date vs price on prediction_date
# ─────────────────────────────────────────────────────────────
def get_true_direction(conn, coin: str,
                       prediction_date: date,
                       target_date: date) -> str | None:
    """
    True direction = UP if target_date price > prediction_date price
                   = DOWN otherwise
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT fetch_date, price
        FROM raw_prices
        WHERE coin = %s
          AND fetch_date IN (%s, %s)
        ORDER BY fetch_date ASC
    """, (coin, prediction_date, target_date))
    rows = cur.fetchall()
    cur.close()

    prices = {row[0]: float(row[1]) for row in rows}

    if prediction_date not in prices or target_date not in prices:
        return None

    price_today    = prices[prediction_date]
    price_tomorrow = prices[target_date]

    return "UP" if price_tomorrow > price_today else "DOWN"


# ─────────────────────────────────────────────────────────────
# BLOCK 3 — COMPUTE ROLLING ACCURACY
# ─────────────────────────────────────────────────────────────
def compute_rolling_accuracy(conn, coin: str,
                              as_of_date: date,
                              window: int = ROLLING_WINDOW) -> float | None:
    """
    Compute accuracy over the last `window` days of errors for a coin.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT is_correct
        FROM errors
        WHERE coin = %s
          AND prediction_date <= %s
        ORDER BY prediction_date DESC
        LIMIT %s
    """, (coin, as_of_date, window))
    rows = cur.fetchall()
    cur.close()

    if not rows:
        return None

    correct = sum(1 for r in rows if r[0])
    return round(correct / len(rows), 4)


# ─────────────────────────────────────────────────────────────
# BLOCK 4 — INSERT INTO errors table
# ─────────────────────────────────────────────────────────────
def insert_error(conn, coin: str, prediction_date: date,
                 target_date: date, predicted_direction: str,
                 true_direction: str, confidence: float,
                 rolling_accuracy: float | None):
    """Insert one error evaluation row."""

    is_correct       = (predicted_direction == true_direction)
    needs_retraining = (
        rolling_accuracy is not None and
        rolling_accuracy < ACCURACY_THRESHOLD
    )

    cur = conn.cursor()
    cur.execute("""
        INSERT INTO errors (
            coin, prediction_date, target_date,
            predicted_direction, true_direction,
            is_correct, confidence,
            rolling_accuracy_30d, needs_retraining
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (coin, prediction_date) DO NOTHING
    """, (
        coin,
        prediction_date,
        target_date,
        predicted_direction,
        true_direction,
        is_correct,
        confidence,
        rolling_accuracy,
        needs_retraining,
    ))
    conn.commit()
    cur.close()
    return is_correct, needs_retraining


# ─────────────────────────────────────────────────────────────
# BLOCK 5 — INSERT RETRAINING ALERT
# ─────────────────────────────────────────────────────────────
def insert_retraining_alert(conn, coin: str, rolling_accuracy: float):
    """Insert a retraining alert into retraining_log."""
    from datetime import datetime

    # Check if alert already exists for today
    cur = conn.cursor()
    cur.execute("""
        SELECT id FROM retraining_log
        WHERE coin = %s
          AND DATE(triggered_at) = %s
          AND retrain_status = 'pending'
    """, (coin, date.today()))
    if cur.fetchone():
        print(f"  ℹ️  Retraining alert already exists for {coin.upper()} today")
        cur.close()
        return

    cur.execute("""
        INSERT INTO retraining_log (
            coin, triggered_at, trigger_reason,
            rolling_accuracy_before, retrain_status
        ) VALUES (%s, %s, %s, %s, 'pending')
    """, (
        coin,
        datetime.now(),
        f"30-day rolling accuracy dropped to {rolling_accuracy*100:.1f}% "
        f"(threshold: {ACCURACY_THRESHOLD*100:.0f}%)",
        rolling_accuracy,
    ))
    conn.commit()
    cur.close()
    print(f"  🚨 Retraining alert inserted for {coin.upper()}!")


# ─────────────────────────────────────────────────────────────
# BLOCK 6 — SIMULATION MODE
# Injects fake historical data to test the full pipeline today
# ─────────────────────────────────────────────────────────────
def run_simulation(conn):
    """
    Simulate 35 days of predictions + prices to test the pipeline.
    Inserts fake raw_prices and predictions for past dates,
    then runs error calculation on all of them.
    """
    import random
    random.seed(42)

    print("\n" + "="*55)
    print("  🧪 SIMULATION MODE — injecting 35 days of fake data")
    print("="*55)

    cur  = conn.cursor()
    base = date.today() - timedelta(days=36)

    for coin in COINS:
        base_price = 68000.0 if coin == "btc" else 1970.0
        print(f"\n  Injecting data for {coin.upper()}...")

        for i in range(36):
            day   = base + timedelta(days=i)
            # Simulate random price walk
            base_price *= (1 + random.uniform(-0.03, 0.03))
            price = round(base_price, 2)

            # Insert into raw_prices (skip if exists)
            cur.execute("""
                INSERT INTO raw_prices (coin, fetch_date, price, volume, fear_greed, vix, spy_return, dxy_return)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (coin, fetch_date) DO NOTHING
            """, (coin, day, price, 40000000000, 50, 18.5, 0.002, -0.001))

            # Insert prediction for next day (skip day 0 — no "yesterday")
            if i < 35:
                direction  = random.choice(["UP", "DOWN"])
                confidence = round(random.uniform(52, 72), 2)
                target     = day + timedelta(days=1)
                cur.execute("""
                    INSERT INTO predictions (coin, prediction_date, target_date, predicted_direction, confidence, model_version)
                    VALUES (%s, %s, %s, %s, %s, 'sim')
                    ON CONFLICT (coin, prediction_date) DO NOTHING
                """, (coin, day, target, direction, confidence))

        conn.commit()
        print(f"  ✅ {coin.upper()} simulation data injected")

    cur.close()
    print("\n  ✅ Simulation data ready — running error tracker...\n")


# ─────────────────────────────────────────────────────────────
# BLOCK 7 — DISPLAY SUMMARY
# ─────────────────────────────────────────────────────────────
def show_summary(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT coin,
               COUNT(*)                                    AS total,
               SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) AS correct,
               ROUND(AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END)*100, 1) AS accuracy_pct,
               MAX(rolling_accuracy_30d)*100               AS latest_rolling_acc,
               SUM(CASE WHEN needs_retraining THEN 1 ELSE 0 END) AS retrain_flags
        FROM errors
        GROUP BY coin
        ORDER BY coin
    """)
    rows = cur.fetchall()
    print(f"\n  {'Coin':<6} {'Total':>6} {'Correct':>8} {'Accuracy':>10} "
          f"{'Rolling 30d':>12} {'Retrain Flags':>14}")
    print(f"  {'-'*62}")
    for row in rows:
        print(f"  {row[0]:<6} {row[1]:>6} {row[2]:>8} {float(row[3]):>9.1f}% "
              f"{float(row[4]) if row[4] else 0:>11.1f}% {row[5]:>14}")

    # Show last 5 errors per coin
    print(f"\n  Last 5 evaluations:")
    cur.execute("""
        SELECT coin, prediction_date, predicted_direction,
               true_direction, is_correct, rolling_accuracy_30d, needs_retraining
        FROM errors
        ORDER BY prediction_date DESC, coin
        LIMIT 10
    """)
    rows = cur.fetchall()
    print(f"  {'Coin':<6} {'Date':<12} {'Pred':>6} {'True':>6} "
          f"{'OK?':>5} {'Roll.Acc':>9} {'Retrain?':>9}")
    print(f"  {'-'*58}")
    for row in rows:
        ok      = "✅" if row[4] else "❌"
        retrain = "🚨" if row[6] else "  "
        roll    = f"{float(row[5])*100:.1f}%" if row[5] else "N/A"
        print(f"  {row[0]:<6} {str(row[1]):<12} {row[2]:>6} {row[3]:>6} "
              f"{ok:>5} {roll:>9} {retrain:>9}")
    cur.close()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", action="store_true",
                        help="Inject fake historical data for testing")
    args = parser.parse_args()

    print("\n" + "="*55)
    print(f"  STEP 4 — Error Tracker → errors table")
    print(f"  Date : {date.today()}")
    print("="*55)

    conn = psycopg2.connect(**DB_CONFIG)
    print("  ✅ Connected to crypto_tracker")

    # Optional simulation
    if args.simulate:
        run_simulation(conn)

    # Process each coin
    total_evaluated = 0

    for coin in COINS:
        print(f"\n{'─'*55}")
        print(f"  📊 {coin.upper()} ERROR EVALUATION")
        print(f"{'─'*55}")

        pending = get_pending_predictions(conn, coin)
        print(f"  Pending predictions to evaluate: {len(pending)}")

        if not pending:
            print(f"  ℹ️  Nothing to evaluate yet for {coin.upper()}")
            print(f"      (true price for target_date not yet in raw_prices)")
            continue

        for row in pending:
            _, _, prediction_date, target_date, predicted_dir, confidence = row

            # Get true direction
            true_dir = get_true_direction(
                conn, coin, prediction_date, target_date
            )
            if true_dir is None:
                print(f"  ⚠️  Could not determine true direction for {prediction_date}")
                continue

            # Compute rolling accuracy BEFORE inserting this row
            rolling_acc = compute_rolling_accuracy(conn, coin, prediction_date)

            # Insert error
            is_correct, needs_retrain = insert_error(
                conn, coin, prediction_date, target_date,
                predicted_dir, true_dir, float(confidence), rolling_acc
            )

            # Re-compute rolling accuracy AFTER insert for display
            rolling_acc_after = compute_rolling_accuracy(conn, coin, target_date)

            status = "✅ CORRECT" if is_correct else "❌ WRONG"
            print(f"\n  Date     : {prediction_date} → {target_date}")
            print(f"  Predicted: {predicted_dir} | True: {true_dir} → {status}")
            print(f"  Rolling 30d accuracy: "
                  f"{f'{rolling_acc_after*100:.1f}%' if rolling_acc_after else 'N/A (not enough data)'}")

            # Retraining alert
            if needs_retrain:
                print(f"  🚨 Accuracy below threshold ({ACCURACY_THRESHOLD*100:.0f}%)!")
                insert_retraining_alert(conn, coin, rolling_acc_after)
            else:
                print(f"  ✅ Accuracy above threshold — no retraining needed")

            total_evaluated += 1

    # Summary
    print(f"\n{'─'*55}")
    print(f"  Total evaluated: {total_evaluated} prediction(s)")
    print(f"\n📊 errors table summary:")
    show_summary(conn)

    conn.close()

    print("\n" + "="*55)
    print("  ✅ Step 4 complete!")
    print("  Next: run retrain.py (Step 6) if retraining was flagged")
    print("="*55)