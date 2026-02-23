# ============================================================
# TAVILY FETCHER — Crypto & Macro News  (v3 — recency fix)
# ============================================================
# FIX vs v2:
#   ✅ FIX 1 — published_date now extracted from Tavily results
#   ✅ FIX 2 — recency_score computed per article (was always 0.3)
#              Score: 1.0 = today, decays linearly to 0.1 at 30 days
#              Unknown date → neutral 0.3 fallback (unchanged behaviour)
# ============================================================

import os
from datetime import datetime, timezone
from dotenv   import load_dotenv
from tavily   import TavilyClient
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────
ENV_PATH = r"C:\Users\tlili\OneDrive\Bureau\Bootcamp\AI-powered-cryptocurrency-market-analysis-and-decision-support-system\news\.env"

load_dotenv(ENV_PATH)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("❌ TAVILY_API_KEY not found in .env file")

client = TavilyClient(api_key=TAVILY_API_KEY)

MAX_RESULTS_PER_QUERY = 5
MAX_CONTENT_LENGTH    = 500


# ─────────────────────────────────────────────────────────────
# SEARCH QUERIES
# ─────────────────────────────────────────────────────────────
QUERIES = {
    "btc": [
        "Bitcoin BTC price market analysis today",
        "Bitcoin institutional investors ETF news today",
        "crypto market sentiment today",
    ],
    "eth": [
        "Ethereum ETH price market analysis today",
        "Ethereum network DeFi staking news today",
        "crypto market sentiment today",
    ],
}

MACRO_QUERIES = [
    "Federal Reserve interest rates inflation today",
    "US dollar DXY crypto market impact today",
    "crypto regulation SEC government news today",
]


# ─────────────────────────────────────────────────────────────
# FIX 2 — RECENCY SCORE
# ─────────────────────────────────────────────────────────────
def compute_recency_score(published_date: str) -> float:
    """
    Convert a published_date string into a recency score [0.1 – 1.0].

    Scoring:
        1.0 = published today
        0.1 = published 30+ days ago (floor)
        0.3 = fallback when date is missing or unparseable

    Decay is linear over 30 days:
        score = max(0.1,  1.0 - (age_days / 30))
    """
    if not published_date:
        return 0.3  # unknown date → neutral fallback

    try:
        # Handle both "2025-01-15T10:30:00Z" and "2025-01-15T10:30:00+00:00"
        pub = datetime.fromisoformat(published_date.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        age_days = max((now - pub).days, 0)
        return round(max(0.1, 1.0 - (age_days / 30)), 4)
    except Exception:
        return 0.3  # unparseable → neutral fallback


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def clean_content(text: str, max_len: int = MAX_CONTENT_LENGTH) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\r", " ").strip()
    return text[:max_len] + "..." if len(text) > max_len else text


def search(query: str, max_results: int = MAX_RESULTS_PER_QUERY) -> list:
    """Run a single Tavily search and return cleaned results."""
    try:
        response = client.search(
            query          = query,
            search_depth   = "basic",
            max_results    = max_results,
            include_answer = False,
        )
        articles = []
        for r in response.get("results", []):
            # FIX 1 + 2: extract published_date and compute recency_score
            published_date = r.get("published_date", "")
            articles.append({
                "title"          : r.get("title",   "No title"),
                "url"            : r.get("url",     ""),
                "content"        : clean_content(r.get("content", "")),
                "score"          : round(r.get("score", 0.0), 4),
                "published_date" : published_date,
                "recency_score"  : compute_recency_score(published_date) if published_date else 0.8,
            })
        return articles
    except Exception as e:
        print(f"  ⚠️  Search failed for '{query}': {e}")
        return []


def deduplicate(articles: list) -> list:
    seen, unique = set(), []
    for a in articles:
        if a["url"] not in seen:
            seen.add(a["url"])
            unique.append(a)
    return unique


# ─────────────────────────────────────────────────────────────
# MAIN FETCHER — parallel queries
# ─────────────────────────────────────────────────────────────
def fetch_news(coin: str) -> dict:
    """
    Fetch coin-specific + macro news in PARALLEL.
    All 6 queries fire simultaneously → ~5-8s instead of ~20s.
    """
    coin = coin.lower()
    if coin not in ["btc", "eth"]:
        raise ValueError("coin must be 'btc' or 'eth'")

    print(f"\n{'='*55}")
    print(f"  TAVILY NEWS FETCHER (parallel) — {coin.upper()}")
    print(f"  Date : {datetime.today().strftime('%Y-%m-%d')}")
    print(f"{'='*55}")

    coin_queries = QUERIES[coin]
    all_queries  = coin_queries + MACRO_QUERIES

    coin_articles  = []
    macro_articles = []

    # ── Run all queries in parallel
    print(f"\n  🔍 Searching all {len(all_queries)} queries in parallel...")
    with ThreadPoolExecutor(max_workers=len(all_queries)) as executor:
        future_to_query = {
            executor.submit(search, q): (q, i < len(coin_queries))
            for i, q in enumerate(all_queries)
        }
        for future in as_completed(future_to_query):
            query, is_coin = future_to_query[future]
            results = future.result()
            print(f"    ✅ '{query}' → {len(results)} articles")
            if is_coin:
                coin_articles.extend(results)
            else:
                macro_articles.extend(results)

    coin_articles  = deduplicate(coin_articles)
    macro_articles = deduplicate(macro_articles)

    all_articles = deduplicate(coin_articles + macro_articles)
    all_articles = sorted(all_articles, key=lambda x: x["score"], reverse=True)

    print(f"\n  ✅ Total articles fetched : {len(all_articles)}")
    print(f"     Coin-specific          : {len(coin_articles)}")
    print(f"     Macro                  : {len(macro_articles)}")

    return {
        "coin"          : coin.upper(),
        "date"          : datetime.today().strftime("%Y-%m-%d"),
        "coin_articles" : coin_articles,
        "macro_articles": macro_articles,
        "all_articles"  : all_articles,
        "total"         : len(all_articles),
    }


def format_for_llm(news: dict, max_articles: int = 8) -> str:
    lines = [
        f"RECENT NEWS — {news['coin']} ({news['date']})",
        "=" * 50,
    ]
    for i, article in enumerate(news["all_articles"][:max_articles], 1):
        lines.append(f"\n[{i}] {article['title']}")
        lines.append(f"    {article['content']}")
        lines.append(f"    Published : {article.get('published_date', 'unknown')}")
        lines.append(f"    Recency   : {article.get('recency_score', 0.3)}")
        lines.append(f"    Source    : {article['url']}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# CLI / QUICK TEST
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch crypto news via Tavily")
    parser.add_argument("--coin", type=str, required=True,
                        choices=["btc", "eth"],
                        help="Coin to fetch news for")
    args = parser.parse_args()

    news = fetch_news(args.coin)

    print(f"\n{'='*55}")
    print(f"  FORMATTED FOR LLM:")
    print(f"{'='*55}")
    print(format_for_llm(news))