# ============================================================
# DAILY PIPELINE — Master Script
# ============================================================
# Runs all tracking steps in order every day:
#   1. daily_fetch.py        — fetch live prices
#   2. predict_and_store.py  — run models + store predictions
#   3. error_tracker.py      — evaluate yesterday's predictions
#   4. retrain.py            — retrain if accuracy alert exists
#
# Scheduled by Windows Task Scheduler to run daily at 00:30
# Run manually: python daily_pipeline.py
# ============================================================

import subprocess
import sys
import os
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────
TRACKING_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR      = os.path.join(TRACKING_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

PYTHON = sys.executable  # uses the same python/conda env

STEPS = [
    ("Step 2 — Daily Fetch",     "daily_fetch.py",        []),
    ("Step 3 — Predict & Store", "predict_and_store.py",  []),
    ("Step 4 — Error Tracker",   "error_tracker.py",      []),
    ("Step 6 — Retrain Check",   "retrain.py",            []),
]


# ─────────────────────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────────────────────
def get_log_path() -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(LOG_DIR, f"pipeline_{today}.log")


def log(message: str, log_path: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    line      = f"[{timestamp}] {message}"
    print(line)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ─────────────────────────────────────────────────────────────
# RUN STEP
# ─────────────────────────────────────────────────────────────
def run_step(name: str, script: str,
             extra_args: list, log_path: str) -> bool:
    script_path = os.path.join(TRACKING_DIR, script)
    cmd         = [PYTHON, script_path] + extra_args

    log(f"{'='*50}", log_path)
    log(f"START : {name}", log_path)
    log(f"{'='*50}", log_path)

    try:
        result = subprocess.run(
            cmd,
            capture_output = True,
            text           = True,
            timeout        = 600,   # 10 min max per step
            encoding       = "utf-8",
            env            = {**os.environ, "PYTHONIOENCODING": "utf-8"},
        )

        # Log stdout
        for line in result.stdout.splitlines():
            log(f"  {line}", log_path)

        # Log stderr if any
        if result.stderr:
            for line in result.stderr.splitlines():
                log(f"  ⚠️  {line}", log_path)

        if result.returncode == 0:
            log(f"✅ DONE : {name}\n", log_path)
            return True
        else:
            log(f"❌ FAILED : {name} (exit code {result.returncode})\n", log_path)
            return False

    except subprocess.TimeoutExpired:
        log(f"❌ TIMEOUT : {name} exceeded 10 minutes\n", log_path)
        return False
    except Exception as e:
        log(f"❌ ERROR : {name} — {e}\n", log_path)
        return False


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log_path = get_log_path()
    start    = datetime.now()

    log("", log_path)
    log("=" * 55, log_path)
    log("  CRYPTO TRACKER — DAILY PIPELINE", log_path)
    log(f"  Date : {start.strftime('%Y-%m-%d %H:%M:%S')}", log_path)
    log("=" * 55, log_path)
    log("", log_path)

    results = {}

    for name, script, args in STEPS:
        success          = run_step(name, script, args, log_path)
        results[name]    = success

        # If fetch or predict fails, stop — no point continuing
        if not success and script in ["daily_fetch.py", "predict_and_store.py"]:
            log(f"🛑 Critical step failed — stopping pipeline", log_path)
            break

    # ── Summary
    elapsed = (datetime.now() - start).seconds
    log("", log_path)
    log("=" * 55, log_path)
    log("  PIPELINE SUMMARY", log_path)
    log("=" * 55, log_path)
    for name, success in results.items():
        status = "✅ OK" if success else "❌ FAILED"
        log(f"  {status}  {name}", log_path)
    log("", log_path)
    log(f"  Total time : {elapsed}s", log_path)
    log(f"  Log saved  : {log_path}", log_path)
    log("=" * 55, log_path)