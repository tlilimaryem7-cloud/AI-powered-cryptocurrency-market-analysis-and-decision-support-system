# ============================================================
# CHATBOT — Intent Detection + Routing  (v3 — all fixes)
# ============================================================
# Changes vs v2:
#   ✅ FIX 1 — LLM-based intent + coin detection replaces regex
#              Handles natural phrasing, no order sensitivity
#              Regex kept as fallback if Groq call fails
#   ✅ FIX 2 — market_overview now passes coin="both" to retrieve()
#              (was hardcoded "btc" — penalized ETH articles unfairly)
# ============================================================

import sys
import os
import re
import json

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
# GREETING DETECTION
# ═══════════════════════════════════════════════════════════════

GREETINGS = [
    "hello", "hi", "hey", "good morning", "good afternoon",
    "good evening", "howdy", "greetings", "salut", "bonjour",
    "sup", "what's up", "whats up", "yo",
]

def is_greeting(question: str) -> bool:
    """Return True if the message is a greeting with no crypto content."""
    q = question.lower().strip().rstrip("!.,?")
    if q in GREETINGS:
        return True
    for g in GREETINGS:
        if q.startswith(g) and len(q) <= len(g) + 10:
            return True
    return False


# ═══════════════════════════════════════════════════════════════
# FOLLOWUP DETECTION
# ═══════════════════════════════════════════════════════════════

FOLLOWUP_PATTERNS = [
    r"^why (do you|did you|is that|so)",
    r"^how (do you|so|come|about)",
    r"^what (about|do you mean|else|makes you)",
    r"^(and|so|but|then|ok|okay|really|interesting)",
    r"^(tell me more|explain|elaborate|go on|continue)",
    r"^(which|what one|how much|how long|since when)",
    r"^(is that|are they|does that|do they)",
    r"^(based on|given that|considering)",
    r"^(you mentioned|you said|earlier you)",
    r"^(same|similar|different|compared)",
    r"^(why|how|what|which|when|where|who)\?*$",
    r"^(what is the|what was the|what are the)",
    r"^(can you|could you|would you)",
]

def is_followup(question: str) -> bool:
    """Return True if question looks like a follow-up with no crypto context."""
    q = question.lower().strip()
    for pattern in FOLLOWUP_PATTERNS:
        if re.search(pattern, q):
            return True
    return False


# ═══════════════════════════════════════════════════════════════
# FIX 1 — LLM-BASED INTENT + COIN DETECTION (with regex fallback)
# ═══════════════════════════════════════════════════════════════

INTENT_CLASSIFIER_PROMPT = """You are an intent classifier for a cryptocurrency chatbot.
Classify the user's message into EXACTLY ONE intent and detect the coin.

INTENTS:
- prediction       → asking about future price direction (will BTC go up?, price target, forecast, give me the BTC price)
- explanation      → asking WHY price is moving (why is ETH dropping?, what caused the rally?)
- market_overview  → general market state (what's happening in crypto?, latest news, market update)
- investment_advice → buy/sell/hold decisions (should I buy BTC?, good time to invest?)
- factual_crypto   → definitions, concepts, rankings (what is DeFi?, top 10 coins, how does staking work?)
- off_topic        → completely unrelated to crypto

COINS:
- btc   → user mentions Bitcoin or BTC
- eth   → user mentions Ethereum or ETH
- both  → user mentions both
- null  → no specific coin mentioned

Return ONLY valid JSON with no explanation, no markdown, no extra text:
{"intent": "<intent>", "coin": "<coin>"}"""


def detect_intent_llm(question: str, history: list = None) -> tuple:
    try:
        # Add previous question as context hint for follow-up questions
        context_hint = ""
        if history:
            last_user_msg = next(
                (m["content"] for m in reversed(history) if m["role"] == "user"),
                None
            )
            if last_user_msg:
                 context_hint = (
                    f"\nIMPORTANT: The previous user message was: '{last_user_msg}'. "
                    f"If the current message is a follow-up with no crypto keywords, "
                    f"inherit the intent from the previous message instead of returning off_topic or factual_crypto."
                )

        response = client.chat.completions.create(
            model    = MODEL,
            messages = [
                {"role": "system", "content": INTENT_CLASSIFIER_PROMPT + context_hint},
                {"role": "user",   "content": question},
            ],
            max_tokens  = 30,
            temperature = 0.0,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if the LLM adds them despite instructions
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()

        result = json.loads(raw)
        intent = result.get("intent", "market_overview")
        coin   = result.get("coin", None)

        # Normalise coin
        if coin in ("null", "", None):
            coin = None

        # Validate intent — reject anything unexpected
        valid_intents = {
            "prediction", "explanation", "market_overview",
            "investment_advice", "factual_crypto", "off_topic"
        }
        if intent not in valid_intents:
            print(f"  ⚠️  LLM returned unknown intent '{intent}', falling back to regex")
            return _detect_intent_regex(question), detect_coin(question)

        print(f"  → LLM intent: {intent} | coin: {coin}")
        return intent, coin

    except Exception as e:
        print(f"  ⚠️  LLM intent detection failed ({e}), falling back to regex")
        return _detect_intent_regex(question), detect_coin(question)


# ─────────────────────────────────────────────────────────────
# REGEX FALLBACK (kept intact from v2)
# ─────────────────────────────────────────────────────────────

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
        r"(best|top|safest|cheapest).*(broker|exchange|platform|app).*(crypto|btc|eth|bitcoin|ethereum)",
        r"(broker|exchange|platform).*(crypto|btc|eth|bitcoin|ethereum).*(best|top|safe|cheap|recommend)",
        r"what (broker|exchange|platform).*(use|buy|trade|invest).*(crypto|btc|eth|bitcoin|ethereum)",
        r"(coinbase|binance|kraken|bybit|kucoin|gemini|bitfinex|okx|robinhood|etoro).*(crypto|btc|eth|review|fee|safe)",
        r"(broker|exchange).*(fee|fees|commission|spread|cost)",
        r"(broker|exchange).*(regulated|regulation|license|safe|legit|trusted)",
        r"(broker|exchange).*(deposit|withdraw|fiat|usd|eur)",
        r"difference between (broker|exchange|cex|dex)",
        r"(cex|dex|centralized|decentralized).*(exchange)",
        r"how to (buy|trade|sell).*(btc|eth|bitcoin|ethereum|crypto)",
        r"where to (buy|trade|sell).*(btc|eth|bitcoin|ethereum|crypto)",
        r"index of (cryptocurrencies|coins|tokens)",
        r"cryptocurrency (index|list|ranking|marketcap)",
    ],
    "market_overview": [
        r"what.*(happening|going on).*(crypto|market|btc|eth|bitcoin|ethereum|news|update|today|situation|status)",
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


def _detect_intent_regex(question: str) -> str:
    """Original regex intent detection — used as fallback only."""
    if not is_crypto_related(question):
        return "off_topic"
    q = question.lower()
    for intent in ["prediction", "explanation", "factual_crypto",
                   "market_overview", "investment_advice"]:
        for pattern in INTENT_PATTERNS[intent]:
            if re.search(pattern, q):
                return intent
    return "market_overview"


# ═══════════════════════════════════════════════════════════════
# POST-PROCESSOR: force numbered points onto new lines
# ═══════════════════════════════════════════════════════════════

def format_llm_response(text: str) -> str:
    """
    Ensure numbered list items (1. 2. 3. ...) are each on their own line.
    """
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
Always end with: ⚠️ This is not afinancial advice."""


PROMPTS = {

    "prediction": SYSTEM_BASE + """

For prediction questions, structure your response as:
📊 ML Signal: State the model's direction and confidence percentage exactly.
📰 Key News: go back to the line and List EXACTLY as numbered points, one per line with a line break between each. Never write them as a paragraph. Example format:
1. First news point here.
2. Second news point here.
3. Third news point here.
✅ Bullish Signals: factors supporting an upward move (max 2-3 bullet points).
⚠️ Risk Factors: factors that could invalidate the prediction (max 2-3 bullet points).
🎯 Short-Term Outlook: 1-2 sentence conclusion based on ML signal + news combined.
⚠️ This is not a financial advice.

Keep total response under 280 words. Be specific — use numbers and percentages from the data.
CRITICAL: Your response must directly and specifically answer the user's question below.""",

    "explanation": SYSTEM_BASE + """

For explanation questions, structure your response as:
🔍 What's Happening: 1-2 sentence direct answer to why the price is moving.
📰 News Drivers: the most relevant recent events causing the move.
📊 Technical Picture: what the model features tell us (RSI, momentum, volatility, macro signals).
🔗 Connections: how the news and technical signals relate to each other.
💡 Summary: 1 sentence takeaway.
⚠️ This is not afinancial advice.

Keep total response under 280 words. Prioritize explaining the cause, not the prediction.
CRITICAL: Your response must directly and specifically answer the user's question below.""",

    "market_overview": SYSTEM_BASE + """

For market overview questions, structure your response as:
🌍 Market Snapshot: current overall crypto market condition in 1-2 sentences.
📰 Top Stories: go back to the line and list as numbered points, one per line, never as a paragraph:
1. First story here.
2. Second story here.
3. Third story here.
📊 Sentiment: what Fear & Greed and macro signals are indicating.
🔮 Near-Term Factors to Watch: 2-3 key events or indicators to monitor.
⚠️ This is not afinancial advice.

Keep total response under 280 words. Focus on breadth — cover both BTC, ETH, and macro context.
CRITICAL: Your response must directly and specifically answer the user's question below.""",

    "investment_advice": SYSTEM_BASE + """

For investment advice questions, structure your response as:
📊 Model Signal: what the ML model predicts for direction and confidence.
📰 Market Context: most relevant news that could affect a buying/selling decision, as numbered points on separate lines.
✅ Arguments For: data-backed reasons supporting the action (max 3).
⚠️ Arguments Against: data-backed risks that argue against the action (max 3).
🎯 Neutral Assessment: a balanced 1-2 sentence view based purely on the data.
⚠️ This is not a financial advice. Always do your own research and consider your risk tolerance.

Keep total response under 300 words. Stay neutral — never recommend a specific action.
CRITICAL: Your response must directly and specifically answer the user's question below.""",

    "factual_crypto": SYSTEM_BASE + """

For factual crypto knowledge questions, answer directly and clearly without a fixed structure.
- Answer the user's SPECIFIC question first and foremost
- Use your knowledge about crypto markets, coins, rankings, and technology
- If listing items (e.g. top 5 coins), use a numbered list with one item per line
- Be specific, accurate, and concise
- Keep the response under 200 words
⚠️ This is not a financial advice.

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
                  live_features: dict | None = None,
                  history: list = None) -> str:
    lines = []

    if history:
        lines += [
            "=" * 50,
            "CONVERSATION HISTORY (last 4 messages)",
            "=" * 50,
        ]
        for msg in history[-4:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"[{role}]: {msg['content'][:300]}")
        lines.append("")

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
            # ── Momentum / Trend (both models)
            "rsi_14"               : "RSI-14",
            "rsi_14_lag1"          : "RSI-14 (lag 1d)",
            "rsi_14_lag3"          : "RSI-14 (lag 3d)",
            "macd_histogram"       : "MACD Histogram",
            "macd_lag1"            : "MACD (lag 1d)",
            "momentum_acceleration": "Momentum Acceleration",
            "bb_pct"               : "Bollinger %B",
            # ── Price
            "price_to_ma7"         : "Price vs MA7",
            "price_to_ma30"        : "Price vs MA30",
            # ── Volatility (coin-specific -- shown only when available)
            "volatility_7d"        : "Volatility 7d",
            "volatility_21d"       : "Volatility 21d",
            "volatility_21d_lag3"  : "Volatility 21d (lag 3d)",
            # ── Macro
            "spy_return"           : "SPY Return",
            "spy_return_ma7"       : "SPY Return MA7",
            "dxy_return_ma7"       : "DXY Return MA7",
            "vix_ma14"             : "VIX MA14",
            # ── Sentiment
            "fear_greed_lag1"      : "Fear & Greed (lag 1d)",
            "fear_greed_lag7"      : "Fear & Greed (lag 7d)",
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
# LLM CALL
# ═══════════════════════════════════════════════════════════════

def build_user_prompt(question: str, context: str) -> str:
    return (
        f"User Question: {question}\n\n"
        f"Data Context:\n{context}\n\n"
        "Please provide your structured analysis based strictly on the data above."
    )

def call_llm(intent: str, question: str, context: str, history: list = None) -> str:
    if history is None:
        history = []

    base_prompt   = PROMPTS.get(intent, PROMPTS["market_overview"])
    system_prompt = (
        base_prompt +
        f'\n\nThe user\'s specific question is: "{question}". '
        "Make sure your response directly and specifically addresses this question. "
        "Do not give a generic answer."
    )
    user_prompt = build_user_prompt(question, context)

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-6:]:
        role = "assistant" if msg["role"] == "assistant" else "user"
        messages.append({"role": role, "content": msg["content"]})
    messages.append({"role": "user", "content": user_prompt})

    try:
        response = client.chat.completions.create(
            model       = MODEL,
            messages    = messages,
            max_tokens  = MAX_TOKENS,
            temperature = TEMPERATURE,
        )
        raw = response.choices[0].message.content
        return format_llm_response(raw)

    except Exception as e:
        return f"❌ LLM call failed: {e}"


# ═══════════════════════════════════════════════════════════════
# MAIN ROUTER
# ═══════════════════════════════════════════════════════════════

def chat(question: str, history: list = None) -> dict:
    if history is None:
        history = []

    from news.tavily_fetcher import fetch_news
    from news.rag            import retrieve
    from live_pipeline       import predict, build_live_features

    print(f"\n{'='*60}")
    print(f"  CHATBOT PIPELINE")
    print(f"{'='*60}")
    print(f"  Question : {question}")

    # ── Greeting short-circuit (before any LLM call)
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

    # ── FIX 1: LLM-based intent + coin detection (regex fallback inside)
    intent, coin = detect_intent_llm(question, history)

    print(f"  Intent   : {intent}")
    print(f"  Coin     : {coin if coin else 'not specified'}")

    # ── Block off-topic
    if intent == "off_topic":
        if history and is_followup(question):
            intent = "explanation"
            print("  → Follow-up detected, routing to explanation ✅")
        else:
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
    if coin == "both" and intent in ["prediction", "investment_advice", "explanation"]:
        coin = "btc"
        print("  → Both coins detected, defaulting to BTC for single-coin intents")

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

            prediction = predict(coin if coin else "btc", features=features_series)
            pred_dir   = prediction["direction"]
            q_lower    = question.lower()

            asks_rising  = any(w in q_lower for w in ["rising", "going up", "pumping", "rally", "surge"])
            asks_falling = any(w in q_lower for w in ["falling", "going down", "dropping", "crashing", "dump"])
            contradiction = (asks_rising and "DOWN" in pred_dir) or (asks_falling and "UP" in pred_dir)

            print("\n⚙️  Fetching news...")
            news     = fetch_news(coin if coin else "btc")
            articles = retrieve(question, news["all_articles"], coin if coin else "btc")

            if contradiction:
                actual_move = "falling" if "DOWN" in pred_dir else "rising"
                question = (
                    f"{question} — Note: the ML model actually predicts {coin.upper()} is {actual_move} "
                    f"(confidence: {prediction['confidence']}%). "
                    f"Please clarify this contradiction to the user and explain the actual direction instead."
                )

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
            # FIX 2: pass "both" so ETH articles are not penalized in keyword_boost
            articles = retrieve(question, deduped, "both", top_n=8)

        elif intent == "investment_advice":
            print("\n⚙️  Running ML prediction for investment context...")
            prediction = predict(coin)
            print("\n⚙️  Fetching news...")
            news     = fetch_news(coin)
            articles = retrieve(question, news["all_articles"], coin)

        elif intent == "factual_crypto":
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
        history       = history,
    )

    print("\n⚙️  Calling LLM...")
    analysis = call_llm(intent, question, context, history)

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
    print("  🤖 CRYPTO ANALYST CHATBOT  (v3)")
    print("  Powered by ML prediction + live news + LLaMA 3.3 70B")
    print("  Type 'exit' or 'quit' to stop")
    print("="*60)

    conversation_history = []

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

        result = chat(question, history=conversation_history)
        display_result(result)

        # Maintain conversation history for follow-up detection
        conversation_history.append({"role": "user",      "content": question})
        conversation_history.append({"role": "assistant", "content": result["analysis"]})