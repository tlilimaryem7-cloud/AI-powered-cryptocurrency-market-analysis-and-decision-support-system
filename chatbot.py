# ============================================================
# CHATBOT — Intent Detection + Routing  (v2 — all fixes)
# ============================================================
# Changes vs v1:
#   ✅ FIX 1 — Greeting detection (hello/hi/hey no longer blocked)
#   ✅ FIX 2 — News formatting: numbered points forced onto new lines
#   ✅ FIX 3 — factual_crypto as 6th intent (no ML/news pipeline)
#   ✅ FIX 4 — User question always injected into system prompt
# ============================================================

import sys
import os
import re

# ─────────────────────────────────────────────────────────────
# PATH SETUP
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

client      = Groq(api_key=GROQ_API_KEY)
MODEL       = "llama-3.3-70b-versatile"
MAX_TOKENS  = 1024
TEMPERATURE = 0.3


# ═══════════════════════════════════════════════════════════════
# FIX 1 — GREETING DETECTION
# ═══════════════════════════════════════════════════════════════

GREETINGS = [
    "hello", "hi", "hey", "good morning", "good afternoon",
    "good evening", "howdy", "greetings", "salut", "bonjour",
    "sup", "what's up", "whats up", "yo",
]

def is_greeting(question: str) -> bool:
    """Return True if the message is a greeting with no crypto content."""
    q = question.lower().strip().rstrip("!.,?")
    # Exact match or starts-with match on short messages
    if q in GREETINGS:
        return True
    for g in GREETINGS:
        if q.startswith(g) and len(q) <= len(g) + 10:
            return True
    return False


# ═══════════════════════════════════════════════════════════════
# INTENT DETECTION
# ═══════════════════════════════════════════════════════════════

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
    # FIX 3 — factual_crypto checked BEFORE market_overview to avoid swallowing
    "factual_crypto": [
        r"top \d+ coin",
        r"best coin",
        r"biggest coin",
        r"largest crypto",
        r"what is (defi|nft|halving|staking|blockchain|altcoin|layer|web3|dao|liquidity)",
        r"what are (altcoins|stablecoins|nfts|layer)",
        r"explain (bitcoin|ethereum|crypto|blockchain|defi|nft|halving|staking)",
        r"difference between",
        r"how does.*(crypto|blockchain|btc|eth|mining|staking|defi)",
        r"what.*mean.*(crypto|blockchain|btc|eth|defi)",
        r"definition of",
        r"history of (bitcoin|ethereum|crypto)",
        r"who (created|invented|founded).*(bitcoin|ethereum|crypto)",
        r"how many (bitcoin|btc|ethereum|eth)",
        r"max supply",
        r"total supply",
        r"part of (defi|crypto|blockchain|ethereum|bitcoin)",
        r"(smart contract|consensus|protocol|layer|node|wallet|seed phrase)",
        r"how (do|does|did).*(work|function)",
        r"what.*(role|purpose).*(crypto|blockchain|defi|btc|eth)",
        r"related to (crypto|blockchain|defi|btc|eth)",
        r"examples of (defi|nft|altcoin|layer|web3|dao|liquidity)",
        r"(highest|biggest|largest|top).*(market.?cap|marketcap)",
        r"market.?cap.*(rank|list|top|highest)",
        r"(rank|ranking).*(coin|crypto|token)",
        r"most valuable (coin|crypto|token)",
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

COIN_PATTERNS = {
    "btc": [r"\bbtc\b", r"\bbitcoin\b", r"\bsatoshi\b"],
    "eth": [r"\beth\b", r"\bethereum\b", r"\bether\b", r"\bdefi\b", r"\bvitalik\b"],
}

CRYPTO_CONTEXT_KEYWORDS = [
    "btc", "eth", "bitcoin", "ethereum", "crypto", "blockchain",
    "defi", "altcoin", "market", "token", "coin", "wallet",
    "trading", "exchange", "bull", "bear", "halving", "etf",
    "mining", "staking", "price", "chart", "rally", "dump",
    "nft", "web3", "layer", "dao", "liquidity", "supply",
]


def detect_coin(question: str) -> str | None:
    q     = question.lower()
    found = []
    for coin, patterns in COIN_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, q):
                found.append(coin)
                break
    if len(found) == 1:
        return found[0]
    if len(found) == 2:
        return "both"
    return None


def is_crypto_related(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in CRYPTO_CONTEXT_KEYWORDS)


def detect_intent(question: str) -> str:
    if not is_crypto_related(question):
        return "off_topic"
    q = question.lower()
    # FIX 3: factual_crypto checked before market_overview
    for intent in ["prediction", "explanation", "factual_crypto",
                   "market_overview", "investment_advice"]:
        for pattern in INTENT_PATTERNS[intent]:
            if re.search(pattern, q):
                return intent
    return "market_overview"


# ═══════════════════════════════════════════════════════════════
# FIX 2 — POST-PROCESSOR: force numbered points onto new lines
# ═══════════════════════════════════════════════════════════════

def format_llm_response(text: str) -> str:
    """
    Ensure numbered list items (1. 2. 3. ...) are each on their own line.
    This fixes cases where the LLM writes them inline as a paragraph.
    """
    # Insert newline before every "N." that follows non-newline content
    text = re.sub(r'(?<!\n)\s*(\d+\.)\s+', r'\n\1 ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# ═══════════════════════════════════════════════════════════════
# PROMPT TEMPLATES
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
📰 Key News: List EXACTLY as numbered points, one per line with a line break between each. Never write them as a paragraph. Example format:
1. First news point here.
2. Second news point here.
3. Third news point here.
✅ Bullish Signals: factors supporting an upward move (max 2-3 bullet points).
⚠️ Risk Factors: factors that could invalidate the prediction (max 2-3 bullet points).
🎯 Short-Term Outlook: 1-2 sentence conclusion based on ML signal + news combined.
⚠️ This is not financial advice.

Keep total response under 280 words. Be specific — use numbers and percentages from the data.
CRITICAL: Your response must directly and specifically answer the user's question below.""",

    # ── EXPLANATION ─────────────────────────────────────────
    "explanation": SYSTEM_BASE + """

For explanation questions, structure your response as:
🔍 What's Happening: 1-2 sentence direct answer to why the price is moving.
📰 News Drivers: the most relevant recent events causing the move.
📊 Technical Picture: what the model features tell us (RSI, momentum, volatility, macro signals).
🔗 Connections: how the news and technical signals relate to each other.
💡 Summary: 1 sentence takeaway.
⚠️ This is not financial advice.

Keep total response under 280 words. Prioritize explaining the cause, not the prediction.
CRITICAL: Your response must directly and specifically answer the user's question below.""",

    # ── MARKET OVERVIEW ─────────────────────────────────────
    "market_overview": SYSTEM_BASE + """

For market overview questions, structure your response as:
🌍 Market Snapshot: current overall crypto market condition in 1-2 sentences.
📰 Top Stories: list as numbered points, one per line, never as a paragraph:
1. First story here.
2. Second story here.
3. Third story here.
📊 Sentiment: what Fear & Greed and macro signals are indicating.
🔮 Near-Term Factors to Watch: 2-3 key events or indicators to monitor.
⚠️ This is not financial advice.

Keep total response under 280 words. Focus on breadth — cover both BTC, ETH, and macro context.
CRITICAL: Your response must directly and specifically answer the user's question below.""",

    # ── INVESTMENT ADVICE ───────────────────────────────────
    "investment_advice": SYSTEM_BASE + """

For investment advice questions, structure your response as:
📊 Model Signal: what the ML model predicts for direction and confidence.
📰 Market Context: most relevant news that could affect a buying/selling decision, as numbered points on separate lines.
✅ Arguments For: data-backed reasons supporting the action (max 3).
⚠️ Arguments Against: data-backed risks that argue against the action (max 3).
🎯 Neutral Assessment: a balanced 1-2 sentence view based purely on the data.
⚠️ This is not financial advice. Always do your own research and consider your risk tolerance.

Keep total response under 300 words. Stay neutral — never recommend a specific action.
CRITICAL: Your response must directly and specifically answer the user's question below.""",

    # ── FIX 3: FACTUAL CRYPTO ───────────────────────────────
    "factual_crypto": SYSTEM_BASE + """

For factual crypto knowledge questions, answer directly and clearly without a fixed structure.
- Answer the user's SPECIFIC question first and foremost
- Use your knowledge about crypto markets, coins, rankings, and technology
- If listing items (e.g. top 5 coins), use a numbered list with one item per line
- Be specific, accurate, and concise
- Keep the response under 200 words
⚠️ This is not financial advice.

CRITICAL: Your response must directly and specifically answer the user's question below.""",
}


# ═══════════════════════════════════════════════════════════════
# OFF-TOPIC RESPONSE
# ═══════════════════════════════════════════════════════════════

OFF_TOPIC_RESPONSES = {
    "default": (
        "I'm a specialized cryptocurrency market analyst. My analysis is limited to:\n\n"
        "  • 📈 Bitcoin (BTC) and Ethereum (ETH) price direction predictions\n"
        "  • 📰 Crypto market news, sentiment, and macro impact\n"
        "  • 🔍 Explaining price movements and market behavior\n"
        "  • 💡 Overall crypto market overviews\n"
        "  • ⚖️  Balanced data-driven perspective before buying or selling\n\n"
        "If you have a question about BTC, ETH, or the broader crypto market, I'm here to help!"
    )
}


# ═══════════════════════════════════════════════════════════════
# CONTEXT BUILDER
# ═══════════════════════════════════════════════════════════════

def build_context(intent: str, coin: str | None,
                  prediction: dict | None,
                  articles: list,
                  live_features: dict | None = None) -> str:
    lines = []

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


# ═══════════════════════════════════════════════════════════════
# LLM CALL — FIX 4: user question injected into system prompt
# ═══════════════════════════════════════════════════════════════

def build_user_prompt(question: str, context: str) -> str:
    return (
        f"User Question: {question}\n\n"
        f"Data Context:\n{context}\n\n"
        "Please provide your structured analysis based strictly on the data above."
    )


def call_llm(intent: str, question: str, context: str) -> str:
    """
    Call Groq LLM.
    FIX 4: Append the user's specific question to the system prompt so
    the LLM never ignores it in favour of the generic structure.
    """
    base_prompt   = PROMPTS.get(intent, PROMPTS["market_overview"])
    # Inject question into system prompt
    system_prompt = (
        base_prompt +
        f'\n\nThe user\'s specific question is: "{question}". '
        "Make sure your response directly and specifically addresses this question. "
        "Do not give a generic answer."
    )
    user_prompt = build_user_prompt(question, context)

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
        raw = response.choices[0].message.content
        # FIX 2: post-process to ensure numbered points are on new lines
        return format_llm_response(raw)

    except Exception as e:
        return f"❌ LLM call failed: {e}"


# ═══════════════════════════════════════════════════════════════
# MAIN ROUTER
# ═══════════════════════════════════════════════════════════════

def chat(question: str) -> dict:
    """
    Main entry point for the chatbot.

    1. Check greeting → short-circuit immediately
    2. Detect intent + coin
    3. Block off-topic
    4. Route to the right pipeline
    5. Build context
    6. Call LLM with correct prompt (question always injected)
    7. Return structured result
    """
    from news.tavily_fetcher import fetch_news
    from news.rag            import retrieve
    from live_pipeline       import predict, build_live_features

    print(f"\n{'='*60}")
    print(f"  CHATBOT PIPELINE")
    print(f"{'='*60}")
    print(f"  Question : {question}")

    # ── FIX 1: Greeting short-circuit
    if is_greeting(question):
        print("  → Greeting detected ✅")
        return {
            "question" : question,
            "intent"   : "greeting",
            "coin"     : None,
            "prediction": None,
            "articles" : [],
            "analysis" : (
                "👋 Hello! I'm CoinTrend AI Analyst.\n\n"
                "I can help you with:\n"
                "  • 📈 BTC & ETH price direction predictions\n"
                "  • 📰 Latest crypto & macro news analysis\n"
                "  • 🔍 Explaining market movements\n"
                "  • 💡 Crypto market overviews\n\n"
                "Ask me anything about Bitcoin, Ethereum, or the crypto market!"
            ),
        }

    # ── Intent + coin detection
    intent = detect_intent(question)
    coin   = detect_coin(question)

    print(f"  Intent   : {intent}")
    print(f"  Coin     : {coin if coin else 'not specified'}")

    # ── Block off-topic
    if intent == "off_topic":
        print("  → Off-topic question blocked ✅")
        return {
            "question"  : question,
            "intent"    : "off_topic",
            "coin"      : None,
            "prediction": None,
            "articles"  : [],
            "analysis"  : OFF_TOPIC_RESPONSES["default"],
        }

    # ── Coin fallback
    if coin is None and intent in ["prediction", "investment_advice", "explanation"]:
        coin = "btc"
        print("  → No coin detected, defaulting to BTC")
    if coin == "both":
        coin = "btc"
        print("  → Both coins detected, defaulting to BTC")

    prediction    = None
    live_features = None
    articles      = []

    # ── Pipeline routing
    try:
        if intent == "prediction":
            print("\n⚙️  Running ML prediction...")
            prediction = predict(coin)
            print("\n⚙️  Fetching news...")
            news     = fetch_news(coin)
            articles = retrieve(question, news["all_articles"], coin)

        elif intent == "explanation":
            print("\n⚙️  Building live features...")
            features_series = build_live_features(coin if coin else "btc")
            live_features   = features_series.to_dict()
            print("\n⚙️  Fetching news...")
            news     = fetch_news(coin if coin else "btc")
            articles = retrieve(question, news["all_articles"], coin if coin else "btc")

        elif intent == "market_overview":
            print("\n⚙️  Fetching news (market overview)...")
            news_btc = fetch_news("btc")
            news_eth = fetch_news("eth")
            all_news = news_btc["all_articles"] + news_eth["all_articles"]
            seen, deduped = set(), []
            for a in all_news:
                if a["url"] not in seen:
                    seen.add(a["url"])
                    deduped.append(a)
            articles = retrieve(question, deduped, "btc", top_n=8)

        elif intent == "investment_advice":
            print("\n⚙️  Running ML prediction for investment context...")
            prediction = predict(coin)
            print("\n⚙️  Fetching news...")
            news     = fetch_news(coin)
            articles = retrieve(question, news["all_articles"], coin)

        elif intent == "factual_crypto":
            # FIX 3: No ML, no Tavily — LLM answers from its own knowledge
            print("\n⚙️  Factual question — skipping ML & news pipeline")
            articles      = []
            prediction    = None
            live_features = None

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

    # ── Build context + call LLM
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
    print("  🤖 CRYPTO ANALYST CHATBOT  (v2)")
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