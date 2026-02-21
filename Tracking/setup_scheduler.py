# ============================================================
# STEP 7 — Task Scheduler Setup
# ============================================================
# Creates a Windows Task Scheduler task that runs
# daily_pipeline.py every day at 00:30 AM automatically.
#
# Run ONCE as Administrator:
#   Right-click PowerShell → "Run as Administrator"
#   python setup_scheduler.py
#
# To remove the task later:
#   python setup_scheduler.py --remove
# ============================================================

import subprocess
import sys
import os
import argparse
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────
TASK_NAME    = "CryptoTrackerDailyPipeline"
TASK_TIME    = "00:30"   # runs at 00:30 AM every day
TRACKING_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON       = sys.executable
SCRIPT       = os.path.join(TRACKING_DIR, "daily_pipeline.py")
LOG_DIR      = os.path.join(TRACKING_DIR, "logs")


# ─────────────────────────────────────────────────────────────
# CREATE TASK
# ─────────────────────────────────────────────────────────────
def create_task():
    """Create the scheduled task using schtasks command."""

    print("\n" + "="*55)
    print("  STEP 7 — Windows Task Scheduler Setup")
    print("="*55)
    print(f"\n  Task name  : {TASK_NAME}")
    print(f"  Script     : {SCRIPT}")
    print(f"  Python     : {PYTHON}")
    print(f"  Schedule   : Daily at {TASK_TIME}")
    print(f"  Log dir    : {LOG_DIR}")

    # Build schtasks command
    # /SC DAILY     = run every day
    # /ST 00:30     = start time
    # /RL HIGHEST   = run with highest privileges
    # /F            = force create (overwrite if exists)
    cmd = [
        "schtasks", "/Create",
        "/TN", TASK_NAME,
        "/TR", f'"{PYTHON}" "{SCRIPT}"',
        "/SC", "DAILY",
        "/ST", TASK_TIME,
        "/RL", "HIGHEST",
        "/F",
    ]

    print(f"\n  Running schtasks command...")
    result = subprocess.run(
        cmd,
        capture_output = True,
        text           = True,
        shell          = True,
    )

    if result.returncode == 0:
        print(f"  ✅ Task '{TASK_NAME}' created successfully!")
        print(f"\n  The pipeline will now run automatically every day at {TASK_TIME}")
    else:
        print(f"  ❌ Failed to create task:")
        print(f"     {result.stderr}")
        print(f"\n  💡 Try running PowerShell as Administrator and run again")
        return False

    return True


# ─────────────────────────────────────────────────────────────
# VERIFY TASK
# ─────────────────────────────────────────────────────────────
def verify_task():
    """Check if the task was created and show its details."""
    cmd    = ["schtasks", "/Query", "/TN", TASK_NAME, "/FO", "LIST"]
    result = subprocess.run(
        cmd, capture_output=True, text=True, shell=True
    )
    if result.returncode == 0:
        print(f"\n  📋 Task details:")
        for line in result.stdout.splitlines():
            if any(k in line for k in ["TaskName", "Status",
                                        "Next Run", "Last Run",
                                        "Schedule Type"]):
                print(f"     {line.strip()}")
    else:
        print(f"  ⚠️  Could not verify task: {result.stderr}")


# ─────────────────────────────────────────────────────────────
# REMOVE TASK
# ─────────────────────────────────────────────────────────────
def remove_task():
    """Remove the scheduled task."""
    cmd    = ["schtasks", "/Delete", "/TN", TASK_NAME, "/F"]
    result = subprocess.run(
        cmd, capture_output=True, text=True, shell=True
    )
    if result.returncode == 0:
        print(f"  ✅ Task '{TASK_NAME}' removed successfully")
    else:
        print(f"  ❌ Failed to remove task: {result.stderr}")


# ─────────────────────────────────────────────────────────────
# MANUAL INSTRUCTIONS (fallback if script fails)
# ─────────────────────────────────────────────────────────────
def print_manual_instructions():
    print("""
  ─────────────────────────────────────────────────
  MANUAL SETUP (if the script fails):
  ─────────────────────────────────────────────────
  1. Press Windows key → search "Task Scheduler" → open it
  2. Click "Create Basic Task" on the right panel
  3. Name: CryptoTrackerDailyPipeline → Next
  4. Trigger: Daily → Next
  5. Start time: 00:30 AM → Next
  6. Action: Start a program → Next
  7. Program/script: paste your Python path:
""")
    print(f"     {PYTHON}")
    print("""
  8. Add arguments:
""")
    print(f'     "{SCRIPT}"')
    print("""
  9. Click Finish
  10. Right-click the task → Properties → 
      General tab → check "Run with highest privileges"
  ─────────────────────────────────────────────────
""")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--remove", action="store_true",
                        help="Remove the scheduled task")
    parser.add_argument("--verify", action="store_true",
                        help="Verify the task exists")
    args = parser.parse_args()

    if args.remove:
        print(f"\n  Removing task '{TASK_NAME}'...")
        remove_task()

    elif args.verify:
        verify_task()

    else:
        # Create task
        success = create_task()

        if success:
            verify_task()
        else:
            print_manual_instructions()

        print("\n" + "="*55)
        print("  DAILY PIPELINE ORDER:")
        print("="*55)
        print("  00:30 AM every day:")
        print("  1. daily_fetch.py        (~1 min)  — fetch prices")
        print("  2. predict_and_store.py  (~2 min)  — run models")
        print("  3. error_tracker.py      (~1 min)  — check errors")
        print("  4. retrain.py            (~10 min) — retrain if needed")
        print(f"\n  📁 Logs saved to: {LOG_DIR}")
        print(f"     One log file per day: pipeline_YYYY-MM-DD.log")
        print("="*55)