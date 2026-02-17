# ============================================================
# CHATBOT — Intent Detection + Routing
# ============================================================
# Brain of the system. Receives user question, detects intent
# and coin, routes to the right pipeline and prompt template.
#
# Intent types:
#   - prediction      : "What will BTC do tomorrow?"
#   - explanation     : "Why is BTC dropping?"
#   - market_overview : "What's happening in crypto?"
#   - investment_advice: "Should I buy ETH?"
#   - off_topic       : anything not crypto-related
#
# Run: python chatbot.py
# ============================================================

import sys
import os
import re

# ─────────────────────────────────────────────────────────────
# PATH SETUP — adjust to your project structure
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = r"C:\Users\tlili\OneDrive\Bureau\Bootcamp\AI-powered-cryptocurrency-market-analysis-and-decision-support-system"
SRC_PATH     = os.path.join(PROJECT_ROOT, "src")

sys.path.append(PROJECT_ROOT)
sys.path.append(SRC_PATH)

from groq import Groq
from dotenv import load_dotenv

ENV_PATH = os.path.join(PROJECT_ROOT, "news", ".env")
load_dotenv(ENV_PATH)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in .env file")

client = Groq(api_key=GROQ_API_KEY)
MODEL       = "llama-3.3-70b-versatile"
MAX_TOKENS  = 1024
TEMPERATURE = 0.3


# ═══════════════════════════════════════════════════════════════
# INTENT DETECTION
# ═══════════════════════════════════════════════════════════════

# ── Keywords for intent classification
INTENT_PATTERNS = {
    "prediction": [
        r"\bwill\b.*(btc|eth|bitcoin|ethereum|crypto)",
        r"(btc|eth|bitcoin|ethereum|crypto).*(will|going|tomorrow|next week|predict|forecast|price)",
        r"\bpredict\b",
        r"\bforecast\b",
        r"price target",
        r"where.*price.*go",
        r"going (up|down)",
    ],
    "explanation": [
        r"\bwhy\b.*(btc|eth|bitcoin|ethereum|crypto|market|price|drop|pump|dump|fall|rise|surge)",
        r"(btc|eth|bitcoin|ethereum|crypto).*(drop|pump|dump|fall|rise|surge|crash|rally)",
        r"\bwhat caused\b",
        r"\bwhat happened\b",
        r"\bexplain\b",
        r"\bbehind\b.*(move|drop|pump|crash|rally)",
        r"reason.*(price|market)",
    ],
    "market_overview": [
        r"what.*(happening|going on).*(crypto|market|btc|eth|bitcoin|ethereum)",
        r"(crypto|bitcoin|btc|eth|ethereum).*(news|update|today|situation|status|market)",
        r"\bmarket overview\b",
        r"\bmarket update\b",
        r"\bcrypto market\b",
        r"how is (the )?(crypto|btc|eth|bitcoin|ethereum) (market|doing)",
        r"latest.*crypto",
        r"crypto.*latest",
    ],
    "investment_advice": [
        r"\bshould i\b.*(buy|sell|invest|hold|trade).*(btc|eth|bitcoin|ethereum|crypto)",
        r"(buy|sell|invest|hold|trade).*(btc|eth|bitcoin|ethereum|crypto).*\b(now|today|good)\b",
        r"\bgood time to buy\b",
        r"\bworth buying\b",
        r"\bworth investing\b",
        r"is it (safe|good|smart|wise) to (buy|invest|hold)",
        r"\bentry point\b",
        r"\bbuy the dip\b",
    ],
}

# ── Keywords for coin detection
COIN_PATTERNS = {
    "btc": [r"\bbtc\b", r"\bbitcoin\b", r"\bsatoshi\b"],
    "eth": [r"\beth\b", r"\bethereum\b", r"\bether\b", r"\bdefi\b", r"\bvitalik\b"],
}

# ── Keywords to confirm topic is crypto-related
CRYPTO_CONTEXT_KEYWORDS = [
    "btc", "eth", "bitcoin", "ethereum", "crypto", "blockchain",
    "defi", "altcoin", "market", "token", "coin", "wallet",
    "trading", "exchange", "bull", "bear", "halving", "etf",
    "mining", "staking", "price", "chart", "rally", "dump",
]


def detect_coin(question: str) -> str | None:
    """
    Detect which coin the question is about.
    Returns 'btc', 'eth', or None if ambiguous/not found.
    """
    q = question.lower()
    found = []
    for coin, patterns in COIN_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, q):
                found.append(coin)
                break

    if len(found) == 1:
        return found[0]
    if len(found) == 2:
        return "both"     # e.g. "compare BTC and ETH"
    return None           # no coin detected


def is_crypto_related(question: str) -> bool:
    """
    Returns True if the question is related to crypto/finance.
    Used to block off-topic questions before intent detection.
    """
    q = question.lower()
    return any(kw in q for kw in CRYPTO_CONTEXT_KEYWORDS)


def detect_intent(question: str) -> str:
    """
    Classify the user's question into one of 5 intent types.

    Returns
    -------
    str : 'prediction' | 'explanation' | 'market_overview'
          | 'investment_advice' | 'off_topic'
    """
    # Step 1 — Block off-topic immediately
    if not is_crypto_related(question):
        return "off_topic"

    # Step 2 — Match intent patterns (priority order matters)
    q = question.lower()
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, q):
                return intent

    # Step 3 — Fallback: crypto-related but no clear intent
    # → treat as market overview (safest default)
    return "market_overview"


# ═══════════════════════════════════════════════════════════════
# PROMPT TEMPLATES — one per intent
# ═══════════════════════════════════════════════════════════════

SYSTEM_BASE = """You are a professional cryptocurrency market analyst.
You ONLY answer questions about Bitcoin (BTC), Ethereum (ETH), and crypto markets.
You are factual, data-driven, and concise.
Never guarantee profits or promise returns.
Always end with: ⚠️ This is not financial advice."""


PROMPTS = {

    # ── PREDICTION ──────────────────────────────────────────
    "prediction": SYSTEM_BASE + """

For prediction questions, structure your response as:
📊 ML Signal: State the model's direction and confidence percentage exactly.
📰 Key News: 2-3 most relevant recent news points that affect this prediction.
✅ Bullish Signals: factors supporting an upward move (max 2-3 bullet points).
⚠️ Risk Factors: factors that could invalidate the prediction (max 2-3 bullet points).
🎯 Short-Term Outlook: 1-2 sentence conclusion based on ML signal + news combined.
⚠️ This is not financial advice.

Keep total response under 280 words. Be specific — use numbers and percentages from the data.""",

    # ── EXPLANATION ─────────────────────────────────────────
    "explanation": SYSTEM_BASE + """

For explanation questions, structure your response as:
🔍 What's Happening: 1-2 sentence direct answer to why the price is moving.
📰 News Drivers: the most relevant recent events causing the move.
📊 Technical Picture: what the model features tell us (RSI, momentum, volatility, macro signals).
🔗 Connections: how the news and technical signals relate to each other.
💡 Summary: 1 sentence takeaway.
⚠️ This is not financial advice.

Keep total response under 280 words. Prioritize explaining the cause, not the prediction.""",

    # ── MARKET OVERVIEW ─────────────────────────────────────
    "market_overview": SYSTEM_BASE + """

For market overview questions, structure your response as:
🌍 Market Snapshot: current overall crypto market condition in 1-2 sentences.
📰 Top Stories: 3-4 most important recent developments (use bullet points).
📊 Sentiment: what Fear & Greed and macro signals are indicating.
🔮 Near-Term Factors to Watch: 2-3 key events or indicators to monitor.
⚠️ This is not financial advice.

Keep total response under 280 words. Focus on breadth — cover both BTC, ETH, and macro context.""",

    # ── INVESTMENT ADVICE ───────────────────────────────────
    "investment_advice": SYSTEM_BASE + """

For investment advice questions, structure your response as:
📊 Model Signal: what the ML model predicts for direction and confidence.
📰 Market Context: most relevant news that could affect a buying/selling decision.
✅ Arguments For: data-backed reasons supporting the action (max 3).
⚠️ Arguments Against: data-backed risks that argue against the action (max 3).
🎯 Neutral Assessment: a balanced 1-2 sentence view based purely on the data.
⚠️ This is not financial advice. Always do your own research and consider your risk tolerance.

Keep total response under 300 words. Stay neutral — never recommend a specific action.""",
}


# ═══════════════════════════════════════════════════════════════
# OFF-TOPIC RESPONSE
# ═══════════════════════════════════════════════════════════════

OFF_TOPIC_RESPONSES = {
    "default": (
        "Thank you for your question. However, this falls outside my area of expertise.\n\n"
        "I am a specialized cryptocurrency market analyst. My analysis is limited to:\n"
        "  • 📈 Bitcoin (BTC) and Ethereum (ETH) price direction predictions\n"
        "  • 📰 Crypto market news, sentiment, and macro impact\n"
        "  • 🔍 Explaining price movements and market behavior\n"
        "  • 💡 Overall crypto market overviews\n"
        "  • ⚖️  Balanced data-driven perspective before buying or selling\n\n"
        "I am not able to assist with topics unrelated to cryptocurrency markets. "
        "If you have a question about BTC, ETH, or the broader crypto market, "
        "I am here to help."
    )
}


# ═══════════════════════════════════════════════════════════════
# CONTEXT BUILDER — per intent
# ═══════════════════════════════════════════════════════════════

def build_context(intent: str, coin: str | None,
                  prediction: dict | None,
                  articles: list,
                  live_features: dict | None = None) -> str:
    """
    Build the context string injected into the LLM prompt.
    Content varies by intent.
    """
    lines = []

    # ── ML Prediction block (for prediction + investment_advice)
    if prediction and intent in ["prediction", "investment_advice"]:
        lines += [
            "=" * 50,
            f"ML MODEL PREDICTION — {prediction['coin']}",
            "=" * 50,
            f"Date       : {prediction['date']}",
            f"Direction  : {prediction['direction']}",
            f"Confidence : {prediction['confidence']}%",
            "",
        ]

    # ── Live feature values (for explanation)
    if live_features and intent == "explanation":
        lines += [
            "=" * 50,
            "TECHNICAL INDICATORS (live)",
            "=" * 50,
        ]
        feature_labels = {
            "rsi_14"              : "RSI-14",
            "macd_histogram"      : "MACD Histogram",
            "momentum_acceleration": "Momentum Acceleration",
            "volatility_21d"      : "Volatility 21d",
            "bb_pct"              : "Bollinger %B",
            "vix_ma14"            : "VIX MA14",
            "vix_regime"          : "VIX Regime (>20 = fear)",
            "fear_greed_ma7"      : "Fear & Greed MA7",
            "bull_bear_flag"      : "Bull/Bear Flag (1=bull)",
            "spy_return_ma7"      : "SPY Return MA7",
            "dxy_return_ma7"      : "DXY Return MA7",
        }
        for key, label in feature_labels.items():
            val = live_features.get(key)
            if val is not None:
                lines.append(f"  {label:<30}: {round(float(val), 4)}")
        lines.append("")

    # ── News articles block (always included except off_topic)
    if articles:
        lines += [
            "=" * 50,
            f"RELEVANT NEWS ({len(articles)} articles)",
            "=" * 50,
        ]
        for i, article in enumerate(articles, 1):
            lines += [
                f"\n[{i}] {article['title']}",
                f"    {article['content']}",
            ]
            if article.get("published_date"):
                lines.append(f"    Published : {article['published_date']}")

    return "\n".join(lines)


def build_user_prompt(question: str, context: str) -> str:
    return (
        f"User Question: {question}\n\n"
        f"Data Context:\n{context}\n\n"
        "Please provide your structured analysis based strictly on the data above."
    )


# ═══════════════════════════════════════════════════════════════
# LLM CALL
# ═══════════════════════════════════════════════════════════════

def call_llm(intent: str, question: str, context: str) -> str:
    """Call Groq LLM with the intent-specific system prompt."""
    system_prompt = PROMPTS.get(intent, PROMPTS["market_overview"])
    user_prompt   = build_user_prompt(question, context)

    try:
        response = client.chat.completions.create(
            model    = MODEL,
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens  = MAX_TOKENS,
            temperature = TEMPERATURE,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ LLM call failed: {e}"


# ═══════════════════════════════════════════════════════════════
# MAIN ROUTER
# ═══════════════════════════════════════════════════════════════

def chat(question: str) -> dict:
    """
    Main entry point for the chatbot.

    1. Detect intent + coin
    2. Block off-topic
    3. Route to the right pipeline
    4. Build context
    5. Call LLM with correct prompt template
    6. Return structured result

    Parameters
    ----------
    question : str — raw user question

    Returns
    -------
    dict with keys:
        question, intent, coin, prediction, articles, analysis
    """
    from news.tavily_fetcher import fetch_news
    from news.rag            import retrieve, format_context
    from live_pipeline       import predict, build_live_features

    print(f"\n{'='*60}")
    print(f"  CHATBOT PIPELINE")
    print(f"{'='*60}")
    print(f"  Question : {question}")

    # ── Step 1: Intent + coin detection
    intent = detect_intent(question)
    coin   = detect_coin(question)

    print(f"  Intent   : {intent}")
    print(f"  Coin     : {coin if coin else 'not specified'}")

    # ── Step 2: Block off-topic
    if intent == "off_topic":
        print("  → Off-topic question blocked ✅")
        return {
            "question" : question,
            "intent"   : "off_topic",
            "coin"     : None,
            "prediction": None,
            "articles" : [],
            "analysis" : OFF_TOPIC_RESPONSES["default"],
        }

    # ── Step 3: Resolve coin fallback
    # If no coin detected, default to BTC for prediction/investment,
    # None is fine for market_overview
    if coin is None and intent in ["prediction", "investment_advice", "explanation"]:
        coin = "btc"
        print(f"  → No coin detected, defaulting to BTC")
    if coin == "both":
        coin = "btc"
        print(f"  → Both coins detected, defaulting to BTC")

    prediction    = None
    live_features = None
    articles      = []

    # ── Step 4: Run pipeline based on intent
    try:
        # PREDICTION — needs ML model + news
        if intent == "prediction":
            print("\n⚙️  Running ML prediction...")
            prediction = predict(coin)
            print("\n⚙️  Fetching news...")
            news       = fetch_news(coin)
            articles   = retrieve(question, news["all_articles"], coin)

        # EXPLANATION — needs live features + news (no prediction direction needed)
        elif intent == "explanation":
            print("\n⚙️  Building live features for explanation...")
            features_series = build_live_features(coin if coin else "btc")
            live_features   = features_series.to_dict()
            print("\n⚙️  Fetching news...")
            news     = fetch_news(coin if coin else "btc")
            articles = retrieve(question, news["all_articles"], coin if coin else "btc")

        # MARKET OVERVIEW — news only, no ML
        elif intent == "market_overview":
            print("\n⚙️  Fetching news (market overview)...")
            # Fetch for both coins to get broader picture
            news_btc  = fetch_news("btc")
            news_eth  = fetch_news("eth")
            all_news  = news_btc["all_articles"] + news_eth["all_articles"]
            # Deduplicate by URL
            seen, deduped = set(), []
            for a in all_news:
                if a["url"] not in seen:
                    seen.add(a["url"])
                    deduped.append(a)
            articles = retrieve(question, deduped, "btc", top_n=8)

        # INVESTMENT ADVICE — needs ML + news
        elif intent == "investment_advice":
            print("\n⚙️  Running ML prediction for investment context...")
            prediction = predict(coin)
            print("\n⚙️  Fetching news...")
            news       = fetch_news(coin)
            articles   = retrieve(question, news["all_articles"], coin)

    except Exception as e:
        print(f"  ⚠️  Pipeline error: {e}")
        return {
            "question"  : question,
            "intent"    : intent,
            "coin"      : coin,
            "prediction": None,
            "articles"  : [],
            "analysis"  : f"❌ Error running analysis pipeline: {e}",
        }

    # ── Step 5: Build context + call LLM
    print("\n⚙️  Building context...")
    context = build_context(
        intent        = intent,
        coin          = coin,
        prediction    = prediction,
        articles      = articles,
        live_features = live_features,
    )

    print("\n⚙️  Calling LLM...")
    analysis = call_llm(intent, question, context)

    print("\n✅ Analysis complete")

    return {
        "question"  : question,
        "intent"    : intent,
        "coin"      : coin,
        "prediction": prediction,
        "articles"  : articles,
        "analysis"  : analysis,
    }


# ═══════════════════════════════════════════════════════════════
# DISPLAY HELPER
# ═══════════════════════════════════════════════════════════════

def display_result(result: dict):
    """Pretty-print the chatbot result."""
    print(f"\n{'='*60}")
    print(f"  CHATBOT RESPONSE")
    print(f"{'='*60}")
    print(f"  Question : {result['question']}")
    print(f"  Intent   : {result['intent']}")
    if result["coin"]:
        print(f"  Coin     : {result['coin'].upper()}")
    if result["prediction"]:
        p = result["prediction"]
        print(f"  ML Signal: {p['direction']} ({p['confidence']}% confidence)")
    print(f"\n{'-'*60}")
    print(result["analysis"])
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════
# CLI — INTERACTIVE LOOP
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  🤖 CRYPTO ANALYST CHATBOT")
    print("  Powered by ML prediction + live news + LLaMA 3.3 70B")
    print("  Type 'exit' or 'quit' to stop")
    print("="*60)

    while True:
        try:
            question = input("\n💬 Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not question:
            continue
        if question.lower() in ["exit", "quit", "q"]:
            print("\n👋 Goodbye!")
            break

        result = chat(question)
        display_result(result)