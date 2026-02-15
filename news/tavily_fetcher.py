# ============================================================
# TAVILY FETCHER — Crypto & Macro News
# ============================================================
# Input  : coin ("btc" or "eth") + optional custom query
# Output : list of relevant articles with title, url,
#          content snippet, and published date
#
# Searches:
#   1. Coin-specific news    (BTC/ETH price, market, sentiment)
#   2. Macro news            (Fed, inflation, regulation)
#   3. Crypto market news    (general crypto market conditions)
# ============================================================

import os
from datetime import datetime
from dotenv   import load_dotenv
from tavily   import TavilyClient

# ─────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────
ENV_PATH = r"C:\Users\tlili\OneDrive\Bureau\Bootcamp\AI-powered-cryptocurrency-market-analysis-and-decision-support-system\news\.env"

load_dotenv(ENV_PATH)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("❌ TAVILY_API_KEY not found in .env file")

client = TavilyClient(api_key=TAVILY_API_KEY)

# Search config
MAX_RESULTS_PER_QUERY = 5     # articles per search query
MAX_CONTENT_LENGTH    = 500   # chars per article snippet


# ─────────────────────────────────────────────────────────────
# SEARCH QUERIES PER COIN
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
# HELPERS
# ─────────────────────────────────────────────────────────────
def clean_content(text: str, max_len: int = MAX_CONTENT_LENGTH) -> str:
    """Truncate and clean article content."""
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\r", " ").strip()
    return text[:max_len] + "..." if len(text) > max_len else text


def search(query: str, max_results: int = MAX_RESULTS_PER_QUERY) -> list:
    """
    Run a single Tavily search and return cleaned results.

    Returns
    -------
    list of dicts with keys: title, url, content, score
    """
    try:
        response = client.search(
            query          = query,
            search_depth   = "basic",
            max_results    = max_results,
            include_answer = False,
        )
        articles = []
        for r in response.get("results", []):
            articles.append({
                "title"  : r.get("title",   "No title"),
                "url"    : r.get("url",     ""),
                "content": clean_content(r.get("content", "")),
                "score"  : round(r.get("score", 0.0), 4),
            })
        return articles
    except Exception as e:
        print(f"  ⚠️  Search failed for '{query}': {e}")
        return []


def deduplicate(articles: list) -> list:
    """Remove duplicate articles by URL."""
    seen = set()
    unique = []
    for a in articles:
        if a["url"] not in seen:
            seen.add(a["url"])
            unique.append(a)
    return unique


# ─────────────────────────────────────────────────────────────
# MAIN FETCHER
# ─────────────────────────────────────────────────────────────
def fetch_news(coin: str) -> dict:
    """
    Fetch coin-specific + macro news for a given coin.

    Parameters
    ----------
    coin : str — "btc" or "eth"

    Returns
    -------
    dict with keys:
        coin          : str
        date          : str
        coin_articles : list of articles
        macro_articles: list of articles
        all_articles  : list of all articles combined
        total         : int
    """
    coin = coin.lower()
    if coin not in ["btc", "eth"]:
        raise ValueError("coin must be 'btc' or 'eth'")

    print(f"\n{'='*55}")
    print(f"  TAVILY NEWS FETCHER — {coin.upper()}")
    print(f"  Date : {datetime.today().strftime('%Y-%m-%d')}")
    print(f"{'='*55}")

    # ── Coin-specific news
    print(f"\n  🔍 Searching {coin.upper()} news...")
    coin_articles = []
    for query in QUERIES[coin]:
        results = search(query)
        print(f"    '{query}' → {len(results)} articles")
        coin_articles.extend(results)
    coin_articles = deduplicate(coin_articles)

    # ── Macro news
    print(f"\n  🔍 Searching macro news...")
    macro_articles = []
    for query in MACRO_QUERIES:
        results = search(query)
        print(f"    '{query}' → {len(results)} articles")
        macro_articles.extend(results)
    macro_articles = deduplicate(macro_articles)

    # ── Combine + sort by relevance score
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
    """
    Format fetched news into a clean string for LLM prompt injection.

    Parameters
    ----------
    news        : dict — output of fetch_news()
    max_articles: int  — max articles to include (top by score)

    Returns
    -------
    str — formatted news context ready for LLM prompt
    """
    lines = [
        f"RECENT NEWS — {news['coin']} ({news['date']})",
        "=" * 50,
    ]

    top_articles = news["all_articles"][:max_articles]

    for i, article in enumerate(top_articles, 1):
        lines.append(f"\n[{i}] {article['title']}")
        lines.append(f"    {article['content']}")
        lines.append(f"    Source: {article['url']}")

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
