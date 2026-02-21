Performance Tracking System — Test Results
AI-Powered Cryptocurrency Market Analysis and Decision Support System
Test Date: 2026-02-21
Tested by: Montassar Tlili

Overview
This document summarizes the end-to-end test of the Model Performance Tracking System — a pipeline designed to detect model degradation and automatically retrain BTC and ETH prediction models when their accuracy drops below an acceptable threshold.

System Architecture
[daily_fetch.py]         → Fetches live prices + macro signals daily
        │
        ▼
[raw_prices table]       → Stores daily BTC/ETH prices, VIX, SPY, DXY, Fear & Greed
        │
        ▼
[predict_and_store.py]   → Runs ML models and stores predictions
        │
        ▼
[predictions table]      → Stores direction (UP/DOWN) + confidence per coin per day
        │
        ▼
[error_tracker.py]       → Compares predictions vs true prices next day
        │
        ▼
[errors table]           → Tracks is_correct + rolling 30-day accuracy
        │
        ▼
If rolling accuracy < 52% → retraining_log (status: pending)
        │
        ▼
[retrain.py]             → Retrains model on fresh data → saves new model
        │
        ▼
[retraining_log table]   → Logs accuracy before/after + model version

Step 1 — Database Setup
Script: db_setup.py
Result: ✅ Success
TableColumnsPurposeraw_prices10Daily live prices + macro signalspredictions8Model predictions with target timestampserrors11Prediction errors + rolling accuracyretraining_log12Retraining history and model performance

Step 2 — Daily Data Fetch
Script: daily_fetch.py
Result: ✅ Success
CoinDatePriceVolumeFear & GreedVIXBTC2026-02-21$68,043.8947,235,649,5367 (Extreme Fear)19.09ETH2026-02-21$1,971.5321,640,841,2167 (Extreme Fear)19.09
Macro Signals:

SPY return: +0.007206 (slight positive)
DXY return: -0.001441 (dollar slightly weaker)
VIX: 19.09 (below 20 — no fear regime)


Step 3 — Live Predictions
Script: predict_and_store.py
Result: ✅ Success
CoinPrediction DateTarget DateDirectionConfidenceBTC2026-02-212026-02-22⬇️ DOWN59.77%ETH2026-02-212026-02-22⬇️ DOWN64.53%

Both models predicted a downward move for the next day, consistent with extreme fear sentiment (Fear & Greed = 7).


Step 4 — Error Tracking (Simulation Test)
Script: error_tracker.py --simulate
Result: ✅ Success
Simulation period: 2026-01-16 → 2026-02-20 (35 days per coin = 70 total evaluations)

⚠️ Note: Simulation used random UP/DOWN predictions to stress-test the pipeline.
Real model accuracy (~73%) is not reflected here — only pipeline functionality is tested.

BTC Results
MetricValueTotal predictions evaluated35Correct predictions18Overall accuracy51.4%Latest rolling 30d accuracy60.0%Retraining flags triggered15
ETH Results
MetricValueTotal predictions evaluated35Correct predictions19Overall accuracy54.3%Latest rolling 30d accuracy60.0%Retraining flags triggered20
Sample Evaluations — Last 10 rows
CoinDatePredictedTrueCorrect?Rolling AccRetrain?BTC2026-02-19DOWNDOWN✅50.0%🚨ETH2026-02-19DOWNUP❌60.0%—BTC2026-02-18DOWNDOWN✅50.0%🚨ETH2026-02-18DOWNUP❌60.0%—BTC2026-02-17UPDOWN❌50.0%🚨ETH2026-02-17UPUP✅60.0%—BTC2026-02-16DOWNUP❌53.3%—ETH2026-02-16UPUP✅56.7%—BTC2026-02-15UPUP✅50.0%🚨ETH2026-02-15UPDOWN❌56.7%—
Rolling Accuracy Threshold Logic
Threshold : 52% (just above random/coin flip baseline of 50%)
Window    : 30 days rolling
Action    : If rolling accuracy < 52% → insert alert into retraining_log

Step 6 — Automatic Retraining
Script: retrain.py
Result: ✅ Success for both coins
Trigger: 2 pending alerts detected from simulation
BTC Retraining
ValueTrigger reason30-day rolling accuracy dropped to 50.0%Training data2018-02-08 → 2022-12-31 (1,788 rows)Validation data2023-01-01 → 2024-12-31 (731 rows)Test data2025-01-01 → 2026-02-20 (416 rows)Old model accuracy73.3%New model accuracy73.8% ✅DecisionNew model savedBackupbtc_model_v_20260221_0034_backup.pklStatus✅ SUCCESS
ETH Retraining
ValueTrigger reason30-day rolling accuracy dropped to 0.0%Training data2018-02-08 → 2022-12-31 (1,788 rows)Validation data2023-01-01 → 2024-12-31 (731 rows)Test data2025-01-01 → 2026-02-20 (416 rows)Old model accuracy69.7%New model accuracy70.2% ✅DecisionNew model savedBackupeth_model_v_20260221_0034_backup.pklStatus✅ SUCCESS
Retraining Log Summary
CoinTriggeredStatusAcc BeforeAcc AfterVersionBTC2026-02-21 00:17:42✅ success50.0%73.8%v_20260221_0034ETH2026-02-21 00:17:42✅ successN/A70.2%v_20260221_0034

Key Design Decisions
DecisionReasonAccuracy threshold at 52%Just above random baseline (50%) — below this the model has no edgeRolling window of 30 daysEnough data to be statistically meaningful, not too slow to reactCompare new vs old before savingNever replace a working model with a worse oneBackup before overwritingAlways possible to roll back to any previous versionOne alert per coin per dayPrevents duplicate alerts from flooding the logtarget_date = prediction_date + 1True price is only available the next day — timestamps are critical

What This System Solves
ProblemSolutionConcept drift — market regimes change over timeRolling accuracy tracker detects degradationModel becomes irrelevant months after trainingAutomatic retraining on fresh dataRetraining makes things worseNew model evaluated before replacing old oneNo history of model changesretraining_log tracks every retraining eventManual monitoring requiredFully automated via Task Scheduler (Step 7)

Files in tracking/
FilePurposedb_setup.pyOne-time database and table creationdaily_fetch.pyDaily price + macro data fetchpredict_and_store.pyDaily prediction using live featureserror_tracker.pyDaily error evaluation + rolling accuracyretrain.pyRetraining triggered by accuracy alertstests/test_simulation.mdThis file — test results and documentation