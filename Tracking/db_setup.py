# ============================================================
# STEP 1 — PostgreSQL Database Setup
# ============================================================
# Creates the database schema for the performance tracking system
#
# Tables:
#   1. raw_prices      — daily live prices fetched from live_pipeline
#   2. predictions     — model predictions with target timestamps
#   3. errors          — prediction errors vs true prices
#   4. retraining_log  — tracks when retraining was triggered + results
#
# Run ONCE to initialize the database.
# Run: python db_setup.py
# ============================================================

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# ─────────────────────────────────────────────────────────────
# SETTINGS — update these to match your local PostgreSQL setup
# ─────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host"    : "localhost",
    "port"    : 5432,
    "user"    : "postgres",        # your PostgreSQL username
    "password": "Mt889933!!",   # your PostgreSQL password
    "dbname"  : "crypto_tracker",  # will be created if it doesn't exist
}


# ─────────────────────────────────────────────────────────────
# STEP 1A — Create the database if it doesn't exist
# ─────────────────────────────────────────────────────────────
def create_database():
    """Connect to default 'postgres' db and create crypto_tracker if needed."""
    conn = psycopg2.connect(
        host     = DB_CONFIG["host"],
        port     = DB_CONFIG["port"],
        user     = DB_CONFIG["user"],
        password = DB_CONFIG["password"],
        dbname   = "postgres",   # connect to default db first
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_CONFIG["dbname"],))
    exists = cur.fetchone()

    if not exists:
        cur.execute(sql.SQL("CREATE DATABASE {}").format(
            sql.Identifier(DB_CONFIG["dbname"])
        ))
        print(f"  ✅ Database '{DB_CONFIG['dbname']}' created")
    else:
        print(f"  ℹ️  Database '{DB_CONFIG['dbname']}' already exists — skipping")

    cur.close()
    conn.close()


# ─────────────────────────────────────────────────────────────
# STEP 1B — Create all tables
# ─────────────────────────────────────────────────────────────
TABLES = {

    # ── TABLE 1: raw_prices
    # Stores daily live prices fetched from live_pipeline
    # One row per coin per day
    "raw_prices": """
        CREATE TABLE IF NOT EXISTS raw_prices (
            id             SERIAL PRIMARY KEY,
            coin           VARCHAR(10)    NOT NULL,          -- 'btc' or 'eth'
            fetch_date     DATE           NOT NULL,          -- date data was fetched
            price          NUMERIC(18, 2) NOT NULL,          -- closing price USD
            volume         BIGINT,                           -- daily volume
            fear_greed     SMALLINT,                         -- 0-100
            vix            NUMERIC(8, 2),                    -- VIX level
            spy_return     NUMERIC(10, 6),                   -- SPY daily log return
            dxy_return     NUMERIC(10, 6),                   -- DXY daily log return
            created_at     TIMESTAMP DEFAULT NOW(),
            UNIQUE (coin, fetch_date)                        -- no duplicates per day
        );
    """,

    # ── TABLE 2: predictions
    # Stores model predictions BEFORE the true price is known
    # prediction_date = today (when we run the model)
    # target_date     = tomorrow (when the true price will be available)
    "predictions": """
        CREATE TABLE IF NOT EXISTS predictions (
            id                SERIAL PRIMARY KEY,
            coin              VARCHAR(10)    NOT NULL,       -- 'btc' or 'eth'
            prediction_date   DATE           NOT NULL,       -- date prediction was made
            target_date       DATE           NOT NULL,       -- date the prediction is FOR
            predicted_direction VARCHAR(4)   NOT NULL,       -- 'UP' or 'DOWN'
            confidence        NUMERIC(5, 2)  NOT NULL,       -- model confidence 0-100%
            model_version     VARCHAR(50)    DEFAULT 'v1',   -- track which model was used
            created_at        TIMESTAMP DEFAULT NOW(),
            UNIQUE (coin, prediction_date)                   -- one prediction per coin per day
        );
    """,

    # ── TABLE 3: errors
    # Filled the next day when true_price becomes available in raw_prices
    # Compares predicted_direction with actual price movement
    "errors": """
        CREATE TABLE IF NOT EXISTS errors (
            id                  SERIAL PRIMARY KEY,
            coin                VARCHAR(10)  NOT NULL,       -- 'btc' or 'eth'
            prediction_date     DATE         NOT NULL,       -- date prediction was made
            target_date         DATE         NOT NULL,       -- date the true price is for
            predicted_direction VARCHAR(4)   NOT NULL,       -- 'UP' or 'DOWN'
            true_direction      VARCHAR(4)   NOT NULL,       -- 'UP' or 'DOWN' (actual)
            is_correct          BOOLEAN      NOT NULL,       -- prediction matched reality?
            confidence          NUMERIC(5,2),                -- confidence on that day
            rolling_accuracy_30d NUMERIC(5,4),              -- rolling 30-day accuracy
            needs_retraining    BOOLEAN      DEFAULT FALSE,  -- flag if threshold breached
            created_at          TIMESTAMP DEFAULT NOW(),
            UNIQUE (coin, prediction_date)
        );
    """,

    # ── TABLE 4: retraining_log
    # Created when rolling error exceeds threshold
    # Tracks retraining history and model performance before/after
    "retraining_log": """
        CREATE TABLE IF NOT EXISTS retraining_log (
            id                      SERIAL PRIMARY KEY,
            coin                    VARCHAR(10)   NOT NULL,  -- 'btc' or 'eth'
            triggered_at            TIMESTAMP     NOT NULL,  -- when alert was raised
            trigger_reason          TEXT,                    -- e.g. '30d accuracy dropped to 47%'
            rolling_accuracy_before NUMERIC(5,4),            -- accuracy that triggered alert
            rolling_accuracy_after  NUMERIC(5,4),            -- accuracy after retraining
            training_data_start     DATE,                    -- start of training window used
            training_data_end       DATE,                    -- end of training window used
            new_model_version       VARCHAR(50),             -- new model version saved
            retrain_status          VARCHAR(20)  DEFAULT 'pending',  -- pending/success/failed
            notes                   TEXT,
            created_at              TIMESTAMP DEFAULT NOW()
        );
    """,
}


def create_tables(conn):
    """Create all 4 tables in the crypto_tracker database."""
    cur = conn.cursor()
    for table_name, ddl in TABLES.items():
        cur.execute(ddl)
        print(f"  ✅ Table '{table_name}' ready")
    conn.commit()
    cur.close()


# ─────────────────────────────────────────────────────────────
# STEP 1C — Create indexes for fast querying
# ─────────────────────────────────────────────────────────────
INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_raw_prices_coin_date    ON raw_prices  (coin, fetch_date);",
    "CREATE INDEX IF NOT EXISTS idx_predictions_coin_date   ON predictions (coin, prediction_date);",
    "CREATE INDEX IF NOT EXISTS idx_predictions_target_date ON predictions (coin, target_date);",
    "CREATE INDEX IF NOT EXISTS idx_errors_coin_date        ON errors      (coin, prediction_date);",
    "CREATE INDEX IF NOT EXISTS idx_errors_needs_retraining ON errors      (coin, needs_retraining);",
]

def create_indexes(conn):
    cur = conn.cursor()
    for idx_sql in INDEXES:
        cur.execute(idx_sql)
    conn.commit()
    cur.close()
    print(f"  ✅ Indexes created")


# ─────────────────────────────────────────────────────────────
# STEP 1D — Test the connection and show table summary
# ─────────────────────────────────────────────────────────────
def verify_setup(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT table_name, 
               (SELECT COUNT(*) FROM information_schema.columns 
                WHERE table_name = t.table_name 
                AND table_schema = 'public') AS col_count
        FROM information_schema.tables t
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)
    rows = cur.fetchall()
    print(f"\n  {'Table':<25} {'Columns':>8}")
    print(f"  {'-'*35}")
    for row in rows:
        print(f"  {row[0]:<25} {row[1]:>8}")
    cur.close()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  STEP 1 — PostgreSQL Setup")
    print("="*55)

    # 1A — Create database
    print("\n📦 Creating database...")
    create_database()

    # 1B — Connect to crypto_tracker and create tables
    print("\n📋 Creating tables...")
    conn = psycopg2.connect(**DB_CONFIG)

    create_tables(conn)

    # 1C — Create indexes
    print("\n⚡ Creating indexes...")
    create_indexes(conn)

    # 1D — Verify
    print("\n🔍 Verifying setup...")
    verify_setup(conn)

    conn.close()

    print("\n" + "="*55)
    print("  ✅ Database setup complete!")
    print("  Next: run daily_pipeline.py (Step 2)")
    print("="*55)