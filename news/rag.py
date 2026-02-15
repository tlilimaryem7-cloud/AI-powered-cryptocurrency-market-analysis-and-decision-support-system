# ============================================================
# RAG — Lightweight Retrieval Augmented Generation
# ============================================================
# No vector DB needed — uses TF-IDF cosine similarity to
# score and select the most relevant articles for the
# user's question before passing to the LLM.
#
# Input  : user question + list of fetched articles
# Output : top N most relevant articles as formatted string
# ============================================================

import re
import math
from collections import Counter


# ─────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────
TOP_N_ARTICLES   = 6     # articles to pass to LLM
MIN_SCORE        = 0.01  # minimum relevance score to include


# ─────────────────────────────────────────────────────────────
# TF-IDF HELPERS
# ─────────────────────────────────────────────────────────────
def tokenize(text: str) -> list:
    """Lowercase, remove punctuation, split into tokens."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 2]


def compute_tf(tokens: list) -> dict:
    """Term frequency for a document."""
    count = Counter(tokens)
    total = len(tokens) if tokens else 1
    return {term: freq / total for term, freq in count.items()}


def compute_idf(documents: list) -> dict:
    """Inverse document frequency across all documents."""
    n    = len(documents)
    idf  = {}
    all_terms = set(t for doc in documents for t in doc)
    for term in all_terms:
        doc_count = sum(1 for doc in documents if term in doc)
        idf[term] = math.log((n + 1) / (doc_count + 1)) + 1
    return idf


def tfidf_vector(tokens: list, idf: dict) -> dict:
    """TF-IDF vector for a document."""
    tf = compute_tf(tokens)
    return {term: tf[term] * idf.get(term, 1.0) for term in tf}


def cosine_similarity(vec1: dict, vec2: dict) -> float:
    """Cosine similarity between two TF-IDF vectors."""
    common = set(vec1.keys()) & set(vec2.keys())
    if not common:
        return 0.0
    dot     = sum(vec1[t] * vec2[t] for t in common)
    norm1   = math.sqrt(sum(v**2 for v in vec1.values()))
    norm2   = math.sqrt(sum(v**2 for v in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


# ─────────────────────────────────────────────────────────────
# KEYWORD BOOSTING
# ─────────────────────────────────────────────────────────────
COIN_KEYWORDS = {
    "btc": ["bitcoin", "btc", "crypto", "blockchain", "halving",
            "etf", "institutional", "satoshi", "mining"],
    "eth": ["ethereum", "eth", "ether", "defi", "staking",
            "smart contract", "layer2", "gas", "vitalik"],
}

MACRO_KEYWORDS = ["fed", "federal", "reserve", "inflation", "interest",
                  "rate", "dollar", "dxy", "sec", "regulation",
                  "recession", "gdp", "cpi", "fomc"]


def keyword_boost(article: dict, coin: str) -> float:
    """
    Extra relevance score based on keyword presence.
    Returns a boost value between 0.0 and 0.3.
    """
    text     = (article["title"] + " " + article["content"]).lower()
    boost    = 0.0
    keywords = COIN_KEYWORDS.get(coin.lower(), []) + MACRO_KEYWORDS

    for kw in keywords:
        if kw in text:
            boost += 0.02

    return min(boost, 0.3)


# ─────────────────────────────────────────────────────────────
# MAIN RAG FUNCTION
# ─────────────────────────────────────────────────────────────
def retrieve(question: str, articles: list, coin: str,
             top_n: int = TOP_N_ARTICLES) -> list:
    """
    Score and retrieve the most relevant articles for a question.

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

    # ── Build corpus (question + all article texts)
    article_texts = [
        tokenize(a["title"] + " " + a["content"])
        for a in articles
    ]
    question_tokens = tokenize(question)

    all_docs = article_texts + [question_tokens]
    idf      = compute_idf(all_docs)

    # ── Query vector
    query_vec = tfidf_vector(question_tokens, idf)

    # ── Score each article
    scored = []
    for i, article in enumerate(articles):
        doc_vec      = tfidf_vector(article_texts[i], idf)
        sim          = cosine_similarity(query_vec, doc_vec)
        boost        = keyword_boost(article, coin)
        tavily_score = article.get("score", 0.0)

        # Combined score: TF-IDF similarity + keyword boost + Tavily relevance
        final_score = (sim * 0.5) + (boost * 0.3) + (tavily_score * 0.2)

        scored.append({**article, "relevance_score": round(final_score, 4)})

    # ── Sort by relevance and filter
    scored = sorted(scored, key=lambda x: x["relevance_score"], reverse=True)
    scored = [a for a in scored if a["relevance_score"] >= MIN_SCORE]

    return scored[:top_n]


# ─────────────────────────────────────────────────────────────
# FORMAT FOR LLM
# ─────────────────────────────────────────────────────────────
def format_context(articles: list, coin: str, prediction: dict) -> str:
    """
    Format retrieved articles + ML prediction into a
    clean context string ready for LLM prompt injection.

    Parameters
    ----------
    articles   : list — output of retrieve()
    coin       : str  — "btc" or "eth"
    prediction : dict — output of live_pipeline.predict()

    Returns
    -------
    str — full context block for LLM
    """
    lines = []

    # ── ML Prediction block
    lines += [
        "=" * 55,
        f"ML MODEL PREDICTION — {prediction['coin']}",
        "=" * 55,
        f"Date       : {prediction['date']}",
        f"Direction  : {prediction['direction']}",
        f"Confidence : {prediction['confidence']}%",
        "",
    ]

    # ── News context block
    lines += [
        "=" * 55,
        f"RELEVANT NEWS CONTEXT ({len(articles)} articles)",
        "=" * 55,
    ]

    for i, article in enumerate(articles, 1):
        lines += [
            f"\n[{i}] {article['title']}",
            f"    {article['content']}",
            f"    Relevance: {article['relevance_score']} | "
            f"Source: {article['url']}",
        ]

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
        print(f"       Relevance: {a['relevance_score']}")