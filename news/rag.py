# ============================================================
# RAG — Lightweight Retrieval Augmented Generation (v3)
# ============================================================
# Improvements over v2:
#   + Crypto relevance filter — articles with no crypto
#     keywords are rejected before scoring. Prevents off-topic
#     articles (e.g. mortgage rates) from being selected.
# ============================================================

import re
import math
from collections import Counter


# ─────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────
TOP_N_ARTICLES = 6
MIN_SCORE      = 0.05


# ─────────────────────────────────────────────────────────────
# CRYPTO RELEVANCE FILTER
# ─────────────────────────────────────────────────────────────
CRYPTO_REQUIRED_KEYWORDS = [
    "bitcoin", "btc", "ethereum", "eth", "crypto",
    "blockchain", "defi", "altcoin", "coinbase", "binance",
    "digital asset", "token", "stablecoin", "web3",
    "cryptocurrency", "decentralized", "halving", "etf",
    "satoshi", "mining", "staking", "on-chain",
]

def is_crypto_relevant(article: dict) -> bool:
    """
    Returns True only if the article contains at least one
    crypto-related keyword in its title or content.
    Blocks off-topic articles (mortgages, weather, etc.)
    from ever entering the RAG scoring pipeline.
    """
    text = (article.get("title", "") + " " + article.get("content", "")).lower()
    return any(kw in text for kw in CRYPTO_REQUIRED_KEYWORDS)


# ─────────────────────────────────────────────────────────────
# TF-IDF HELPERS
# ─────────────────────────────────────────────────────────────
def tokenize(text: str) -> list:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 2]


def compute_tf(tokens: list) -> dict:
    count = Counter(tokens)
    total = len(tokens) if tokens else 1
    return {term: freq / total for term, freq in count.items()}


def compute_idf(documents: list) -> dict:
    n         = len(documents)
    all_terms = set(t for doc in documents for t in doc)
    idf       = {}
    for term in all_terms:
        doc_count = sum(1 for doc in documents if term in doc)
        idf[term] = math.log((n + 1) / (doc_count + 1)) + 1
    return idf


def tfidf_vector(tokens: list, idf: dict) -> dict:
    tf = compute_tf(tokens)
    return {term: tf[term] * idf.get(term, 1.0) for term in tf}


def cosine_similarity(vec1: dict, vec2: dict) -> float:
    common = set(vec1.keys()) & set(vec2.keys())
    if not common:
        return 0.0
    dot   = sum(vec1[t] * vec2[t] for t in common)
    norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v**2 for v in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


# ─────────────────────────────────────────────────────────────
# KEYWORD BOOSTING
# ─────────────────────────────────────────────────────────────
COIN_KEYWORDS = {
    "btc": ["bitcoin", "btc", "blockchain", "halving",
            "etf", "institutional", "satoshi", "mining"],
    "eth": ["ethereum", "eth", "ether", "defi", "staking",
            "smart contract", "layer2", "gas", "vitalik"],
}

MACRO_KEYWORDS = [
    "fed", "federal reserve", "inflation", "interest rate",
    "dollar", "dxy", "sec", "regulation", "recession",
    "gdp", "cpi", "fomc", "rate cut", "rate hike",
]


def keyword_boost(article: dict, coin: str) -> float:
    """
    Coin-specific boost + macro boost only if coin also present.
    Max boost capped at 0.25.
    """
    text      = (article["title"] + " " + article["content"]).lower()
    boost     = 0.0

    # Coin-specific boost
    coin_hits = sum(1 for kw in COIN_KEYWORDS.get(coin.lower(), []) if kw in text)
    boost    += min(coin_hits * 0.03, 0.15)

    # Macro boost — only if coin also present
    coin_present = any(kw in text for kw in COIN_KEYWORDS.get(coin.lower(), []))
    if coin_present:
        macro_hits = sum(1 for kw in MACRO_KEYWORDS if kw in text)
        boost     += min(macro_hits * 0.02, 0.10)

    return min(boost, 0.25)


# ─────────────────────────────────────────────────────────────
# MAIN RAG FUNCTION
# ─────────────────────────────────────────────────────────────
def retrieve(question: str, articles: list, coin: str,
             top_n: int = TOP_N_ARTICLES) -> list:
    """
    Score and retrieve the most relevant articles for a question.

    Scoring formula:
        final = (tfidf_sim * 0.50)
              + (recency   * 0.20)
              + (kw_boost  * 0.20)
              + (tavily_sc * 0.10)

    Parameters
    ----------
    question : str  — user's question
    articles : list — output of fetch_news()["all_articles"]
    coin     : str  — "btc" or "eth"
    top_n    : int  — number of articles to return

    Returns
    -------
    list of top_n most relevant articles with relevance_score added
    """
    if not articles:
        return []

    # ── Step 0: Filter out non-crypto articles
    before   = len(articles)
    articles = [a for a in articles if is_crypto_relevant(a)]
    filtered = before - len(articles)
    if filtered > 0:
        print(f"  🚫 RAG filtered {filtered} non-crypto article(s)")

    if not articles:
        return []

    # ── Build corpus
    article_texts   = [tokenize(a["title"] + " " + a["content"]) for a in articles]
    question_tokens = tokenize(question)
    all_docs        = article_texts + [question_tokens]
    idf             = compute_idf(all_docs)

    # ── Query vector
    query_vec = tfidf_vector(question_tokens, idf)

    # ── Score each article
    scored = []
    for i, article in enumerate(articles):
        doc_vec   = tfidf_vector(article_texts[i], idf)
        tfidf_sim = cosine_similarity(query_vec, doc_vec)
        kw_boost  = keyword_boost(article, coin)
        recency   = article.get("recency_score", 0.3)
        tavily_sc = article.get("score", 0.0)

        final_score = (
            (tfidf_sim * 0.50) +
            (recency   * 0.20) +
            (kw_boost  * 0.20) +
            (tavily_sc * 0.10)
        )

        scored.append({
            **article,
            "relevance_score": round(final_score, 4),
            "score_breakdown": {
                "tfidf"  : round(tfidf_sim, 4),
                "recency": round(recency,   4),
                "keyword": round(kw_boost,  4),
                "tavily" : round(tavily_sc, 4),
            }
        })

    # ── Sort and filter
    scored = sorted(scored, key=lambda x: x["relevance_score"], reverse=True)
    scored = [a for a in scored if a["relevance_score"] >= MIN_SCORE]

    return scored[:top_n]


# ─────────────────────────────────────────────────────────────
# FORMAT FOR LLM
# ─────────────────────────────────────────────────────────────
def format_context(articles: list, coin: str, prediction: dict) -> str:
    """Format retrieved articles + ML prediction for LLM prompt."""
    lines = []

    lines += [
        "=" * 55,
        f"ML MODEL PREDICTION — {prediction['coin']}",
        "=" * 55,
        f"Date       : {prediction['date']}",
        f"Direction  : {prediction['direction']}",
        f"Confidence : {prediction['confidence']}%",
        "",
    ]

    lines += [
        "=" * 55,
        f"RELEVANT NEWS CONTEXT ({len(articles)} articles)",
        "=" * 55,
    ]

    for i, article in enumerate(articles, 1):
        lines += [
            f"\n[{i}] {article['title']}",
            f"    {article['content']}",
        ]
        if article.get("published_date"):
            lines.append(f"    Published  : {article['published_date']}")
        lines.append(
            f"    Relevance  : {article['relevance_score']} | "
            f"Source: {article['url']}"
        )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.append(
        r"C:\Users\tlili\OneDrive\Bureau\Bootcamp"
        r"\AI-powered-cryptocurrency-market-analysis-and-decision-support-system"
    )
    from news.tavily_fetcher import fetch_news

    coin     = "btc"
    question = "What will Bitcoin do tomorrow based on market conditions?"

    print("Fetching news...")
    news = fetch_news(coin)

    print("\nRetrieving relevant articles...")
    top_articles = retrieve(question, news["all_articles"], coin)

    print(f"\nTop {len(top_articles)} relevant articles:")
    for i, a in enumerate(top_articles, 1):
        print(f"  [{i}] {a['title']}")
        print(f"       Published : {a.get('published_date', 'unknown')}")
        print(f"       Relevance : {a['relevance_score']}")
        b = a.get("score_breakdown", {})
        print(f"       Breakdown → tfidf: {b.get('tfidf')} | "
              f"recency: {b.get('recency')} | "
              f"keyword: {b.get('keyword')} | "
              f"tavily: {b.get('tavily')}")