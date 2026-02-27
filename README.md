# CoinTrend AI 🪙
### AI-Powered Cryptocurrency Market Analysis & Decision Support System

> A complete end-to-end system that combines machine learning price predictions with real-time news retrieval to explain crypto market behaviour and answer user questions through an intelligent chatbot.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Pipeline](#data-pipeline)
- [ML Models](#ml-models)
- [RAG Chatbot](#rag-chatbot)
- [Performance Tracking](#performance-tracking)
- [Dashboard](#dashboard)
- [Results](#results)

---

## 🎯 Project Overview

Cryptocurrency markets are driven by both **technical signals** (RSI, MACD, volatility) and **unpredictable world events** (ETF approvals, interest rate cuts, geopolitical news). No single tool combines both sources of information effectively.

**CoinTrend AI** solves this by building a system that:

1. Uses **historical + live market data** and **ML/DL models** to predict the future price direction of BTC and ETH
2. Retrieves the **latest macro-economic and crypto news** from the web
3. Combines both to **explain market behaviour** and answer user questions via a chatbot
4. Provides a **real-time Streamlit dashboard** with charts, predictions, and the chatbot
5. **Automatically monitors and retrains** models when performance degrades

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        DATA LAYER                           │
│  yfinance (BTC, ETH, SPY, DXY, VIX) + alternative.me (F&G) │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                   FEATURE PIPELINE                          │
│  preprocessing → 20 SHAP-selected features per coin        │
└───────────────────────┬─────────────────────────────────────┘
                        │
          ┌─────────────┴──────────────┐
          │                            │
┌─────────▼──────────┐    ┌────────────▼───────────┐
│   BTC MODEL v2     │    │   ETH MODEL v2          │
│   Gradient         │    │   Gradient Boosting     │
│   Boosting         │    │   (Optuna-tuned)        │
│   80.1% accuracy   │    │   73.5% accuracy        │
└─────────┬──────────┘    └────────────┬────────────┘
          └─────────────┬──────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                     CHATBOT LAYER                           │
│  Intent Detection → ML Prediction + Tavily News → RAG      │
│  → LLaMA 3.3 70B (Groq) → Structured Response              │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                   STREAMLIT DASHBOARD                       │
│  Price Charts | Predictions | Technical Indicators | Chat  │
└─────────────────────────────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│              PERFORMANCE TRACKING (PostgreSQL)              │
│  raw_prices → predictions → errors → retraining_log        │
│  Rolling 30d accuracy → Auto-retrain if accuracy < 52%     │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### 📈 Price Prediction
- Binary classification: **UP / DOWN** for next trading day
- BTC: GradientBoostingClassifier (Optuna-tuned), 20 SHAP-selected features
- ETH: GradientBoostingClassifier (Optuna-tuned), 20 SHAP-selected features (different set)
- Separate model configs, hyperparameters, and feature lists per coin
- Live features computed on-the-fly from yfinance

### 🤖 AI Chatbot
- 6 intent types: `prediction`, `explanation`, `market_overview`, `investment_advice`, `factual_crypto`, `off_topic`
- Combines ML prediction + live Tavily news articles
- TF-IDF RAG system with keyword boosting and recency scoring
- Powered by LLaMA 3.3 70B via Groq API (free tier)

### 📊 Streamlit Dashboard
- Coin selector (BTC / ETH)
- Historical price chart
- Real-time technical indicators (RSI, MACD, Fear & Greed, VIX)
- Prediction chart with confidence %
- Embedded chatbot panel

### 🔧 Performance Tracking
- Daily automated pipeline via Windows Task Scheduler
- PostgreSQL database with 4 tables
- Rolling 30-day accuracy monitoring
- Automatic retraining when accuracy drops below threshold
- Model versioning and backup

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.13 |
| Data | yfinance, alternative.me API, Tavily API |
| ML | scikit-learn, XGBoost |
| LLM | Groq API (LLaMA 3.3 70B) |
| Database | PostgreSQL 18, psycopg2 |
| Dashboard | Streamlit |
| Automation | Windows Task Scheduler |
| Environment | Anaconda |

---

## 📁 Project Structure

```
AI-powered-cryptocurrency-market-analysis-and-decision-support-system/
│
├── data/
│   ├── raw/
│   │   └── crypto_raw.csv              # Raw merged data
│   └── processed/
│       └── crypto_features.csv         # Engineered features incl. lag features
│
├── models/
│   └── saved_models/
│       ├── btc_model_v2.pkl            # BTC Gradient Boosting (Optuna-tuned)
│       ├── btc_features_v2.pkl         # BTC SHAP-selected feature list
│       ├── btc_metadata_v2.json        # BTC model metadata
│       ├── eth_model_v2.pkl            # ETH Gradient Boosting (Optuna-tuned)
│       ├── eth_features_v2.pkl         # ETH SHAP-selected feature list
│       ├── eth_metadata_v2.json        # ETH model metadata
│       └── backups/                    # Versioned model backups (incl. v1)
│
├── src/
│   ├── pipeline.py                     # Full data + feature pipeline
│   └── live_pipeline.py                # Live feature builder for inference
│
├── news/
│   ├── tavily_fetcher.py               # Parallel news fetching via Tavily
│   ├── rag.py                          # TF-IDF RAG retrieval system
│   ├── llm.py                          # Groq LLM integration
│   └── .env                            # API keys (not tracked in git)
│
├── tracking/                           # Performance tracking system
│   ├── db_setup.py                     # One-time DB + table creation
│   ├── daily_fetch.py                  # Step 2: fetch prices → raw_prices
│   ├── predict_and_store.py            # Step 3: predict → predictions table
│   ├── error_tracker.py                # Step 4: evaluate → errors table
│   ├── retrain.py                      # Step 6: auto-retrain on alert
│   ├── daily_pipeline.py               # Master script (runs all steps)
│   ├── setup_scheduler.py              # Windows Task Scheduler setup
│   ├── logs/                           # Daily pipeline logs
│   └── tests/
│       └── test_simulation.md          # Simulation test results
│
├── chatbot.py                          # Main chatbot router
├── app.py                              # Streamlit dashboard
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/AI-powered-cryptocurrency-market-analysis.git
cd AI-powered-cryptocurrency-market-analysis
```

### 2. Create conda environment
```bash
conda create -n cointrend python=3.13
conda activate cointrend
```

### 3. Install dependencies
```bash
pip install yfinance pandas numpy scikit-learn xgboost joblib
pip install groq tavily-python python-dotenv requests
pip install streamlit psycopg2-binary
```

### 4. Set up API keys
Create a `.env` file inside the `news/` folder:
```
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

Get free API keys at:
- Groq: https://console.groq.com
- Tavily: https://tavily.com

### 5. Set up PostgreSQL
Install PostgreSQL from https://www.postgresql.org/download/windows/

Then run:
```bash
pip install psycopg2-binary
cd tracking
python db_setup.py
```

---

## 🚀 Usage

### Run the full data pipeline
```bash
python src/pipeline.py
```

### Run the Streamlit dashboard
```bash
streamlit run app.py
```

### Run the chatbot (CLI)
```bash
python chatbot.py
```

### Run the performance tracking pipeline manually
```bash
cd tracking
python daily_pipeline.py
```

### Set up automated daily tracking
```bash
# Open PowerShell as Administrator
cd tracking
python setup_scheduler.py
```

### Force retrain a model
```bash
cd tracking
python retrain.py --force btc
python retrain.py --force eth
python retrain.py --force both
```

---

## 📥 Data Pipeline

### Sources
| Source | Data | Frequency |
|---|---|---|
| yfinance | BTC-USD, ETH-USD prices + volume | Daily |
| yfinance | SPY (S&P500), DX-Y.NYB (DXY), ^VIX | Daily |
| alternative.me | Fear & Greed Index (0–100) | Daily |
| Tavily API | Crypto + macro news articles | Live |

### Feature Engineering

All features are computed coin-by-coin. After full engineering, each model uses its own **20 SHAP-selected features** (different sets for BTC and ETH).

#### BTC — 20 Features
| Category | Features |
|---|---|
| Trend | price_to_ma7, price_to_ma30 |
| Momentum | rsi_14, rsi_14_lag1, rsi_14_lag2, rsi_14_lag3, macd_histogram, macd_lag1, macd_lag3, momentum_acceleration, bb_pct |
| Volatility | volatility_7d |
| Price Lags | price_lag1, price_lag3 |
| Volume | volume_lag1 |
| Macro | spy_return, spy_return_ma7, dxy_return_ma7 |
| Sentiment | fear_greed_lag1, fear_greed_lag7 |

#### ETH — 20 Features
| Category | Features |
|---|---|
| Trend | price_to_ma7, price_to_ma30 |
| Momentum | rsi_14, rsi_14_lag1, rsi_14_lag3, macd_histogram, macd_lag1, macd_lag3, momentum_acceleration, bb_pct |
| Volatility | volatility_7d, volatility_21d, volatility_21d_lag3 |
| Price Lags | price_lag1, price_lag3 |
| Macro | spy_return, spy_return_ma7, vix_ma14 |
| Sentiment | fear_greed_lag1, fear_greed_lag7 |

#### Lag Features (shared, computed before subsetting)
`rsi_14_lag1/2/3`, `macd_lag1/3`, `volume_lag1`, `price_lag1/3`, `volatility_21d_lag3`

**Target:** `log_return_1d > 0` → 1 (UP) else 0 (DOWN)

---

## 🤖 ML Models

### Data Split — S2 Regime-Aware
| Split | Period | Description |
|---|---|---|
| Train | 2017 → 2022 | Bear + Bull + Crypto Winter |
| Validation | 2023 → 2024 | Recovery + 2024 Bull Run |
| Test | 2025 → 2026 | Institutional Era |

### BTC — GradientBoostingClassifier v2 (Optuna-tuned)
```
GradientBoostingClassifier(
    n_estimators=349, max_depth=4,
    learning_rate=0.01438, min_samples_leaf=75,
    subsample=0.621
)
Features : 20 SHAP-selected
```

### ETH — GradientBoostingClassifier v2 (Optuna-tuned)
```
GradientBoostingClassifier(
    n_estimators=437, max_depth=6,
    learning_rate=0.01777, min_samples_leaf=97,
    subsample=0.507
)
Features : 20 SHAP-selected (different set from BTC)
```

### Results
| Model | Test Accuracy | MCC | AUC | Train/Test Gap |
|---|---|---|---|---|
| BTC v2 | **80.1%** | 0.608 | 0.874 | 0.09 ✅ |
| ETH v2 | **73.5%** | 0.473 | 0.821 | 0.08 ✅ |

---

## 💬 RAG Chatbot

### Pipeline
```
User Question
    ↓
Intent Detection (6 types)
    ↓
ML Prediction (live features → UP/DOWN + confidence%)
    ↓
Tavily News Fetch (6 parallel queries ~5-8s)
    ↓
RAG Retrieval (TF-IDF cosine similarity + keyword boost + recency)
    ↓
LLaMA 3.3 70B via Groq → Structured Analysis
```

### Intent Types
| Intent | Description | Uses ML | Uses News |
|---|---|---|---|
| `prediction` | Price direction questions | ✅ | ✅ |
| `explanation` | Why is price moving | ✅ | ✅ |
| `market_overview` | General market state | ❌ | ✅ |
| `investment_advice` | Should I buy/sell | ✅ | ✅ |
| `factual_crypto` | What is DeFi/halving/etc | ❌ | ❌ |
| `off_topic` | Non-crypto questions | ❌ | ❌ |

### RAG Scoring Formula
```
final_score = (tfidf_similarity × 0.50)
            + (recency_score    × 0.20)
            + (keyword_boost    × 0.20)
            + (tavily_score     × 0.10)
```

---

## 📊 Performance Tracking

### Database Schema (PostgreSQL)

```sql
raw_prices      -- Daily prices + macro signals (10 cols)
predictions     -- Model outputs + target timestamps (8 cols)
errors          -- is_correct + rolling_accuracy_30d (11 cols)
retraining_log  -- Alert history + before/after accuracy (12 cols)
```

### Auto-Retraining Pipeline
```
Daily at 00:30 AM (Task Scheduler):
  1. daily_fetch.py        → raw_prices table
  2. predict_and_store.py  → predictions table
  3. error_tracker.py      → errors table + rolling accuracy
  4. retrain.py            → retrains if accuracy < 52%
```

### Key Design Decisions
| Decision | Reason |
|---|---|
| Threshold: 52% | Just above random (50%) baseline |
| Window: 30 days | Statistically meaningful, not too slow |
| New vs old comparison | Never replace a good model with a worse one |
| Automatic backup | Always possible to roll back |
| One alert per day | No duplicate alerts |

---

## 📈 Results Summary

| Component | Result |
|---|---|
| BTC Model Accuracy (v2) | **80.1%** on 2025–2026 test set |
| BTC MCC / AUC | 0.608 / 0.874 |
| ETH Model Accuracy (v2) | **73.5%** on 2025–2026 test set |
| ETH MCC / AUC | 0.473 / 0.821 |
| Chatbot Response Time | ~5–8 seconds (parallel news fetch) |
| Daily Pipeline Runtime | ~14 seconds |
| Database Size | ~3.5 MB per coin per 5 years |

---

## ⚠️ Disclaimer

This system is for **educational and research purposes only**.
All predictions and analyses are **not financial advice**.
Always do your own research before making any investment decisions.

---

## 👤 Author

**Maryem Tlili**
Data Science & Machine learning Bootcamp — 2026
