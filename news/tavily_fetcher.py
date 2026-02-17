# ============================================================
# TAVILY FETCHER — Crypto & Macro News (v4)
# ============================================================
# Improvements over v3:
#   + Evergreen domain detection — sites like coinmarketcap,
#     feargreedmeter, alternative.me update live but never
#     show a publication date. These now get recency score 0.6
#     (mid-high) instead of 0.3 neutral — because their content
#     IS current, just undated.
# ============================================================

import os
import re
from datetime import datetime, timedelta
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

MAX_RESULTS_PER_QUERY = 5
MAX_CONTENT_LENGTH    = 1000
DAYS_BACK             = 7

# Live-updating pages — always current but never show a date
EVERGREEN_DOMAINS = [
    "feargreedmeter.com",
    "coinmarketcap.com",
    "alternative.me",
    "coinbase.com",
    "stockgeist.ai",
    "coingecko.com",
    "tradingview.com",
]


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
# DATE RESOLUTION CHAIN
# ─────────────────────────────────────────────────────────────
MONTHS = {
    "january": "01",  "february": "02", "march": "03",
    "april": "04",    "may": "05",      "june": "06",
    "july": "07",     "august": "08",   "september": "09",
    "october": "10",  "november": "11", "december": "12",
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "jun": "06", "jul": "07", "aug": "08", "sep": "09",
    "oct": "10", "nov": "11", "dec": "12",
}


def parse_published_date(raw: str | None) -> str | None:
    """Step 1 — Parse Tavily's own published_date field."""
    if not raw:
        return None
    try:
        for fmt in ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"]:
            try:
                return datetime.strptime(raw[:19], fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        return raw[:10]
    except Exception:
        return None


def extract_date_from_url(url: str) -> str | None:
    """Step 2 — Extract date from URL structure."""
    if not url:
        return None
    m = re.search(
        r"[/\-](20\d{2})[/\-](0[1-9]|1[0-2])[/\-](0[1-9]|[12]\d|3[01])", url
    )
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.search(r"(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])", url)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def extract_date_from_content(content: str) -> str | None:
    """Step 3 — Extract date from article content text."""
    if not content:
        return None
    text = content.lower()

    # ISO: 2026-02-17
    m = re.search(r"(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])", text)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # "February 17, 2026" / "Feb 17, 2026"
    m = re.search(
        r"(january|february|march|april|may|june|july|august|september|"
        r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
        r"\.?\s+(\d{1,2}),?\s+(20\d{2})", text
    )
    if m:
        return f"{m.group(3)}-{MONTHS.get(m.group(1), '01')}-{m.group(2).zfill(2)}"

    # "17 February 2026"
    m = re.search(
        r"(\d{1,2})\s+(january|february|march|april|may|june|july|august|"
        r"september|october|november|december|jan|feb|mar|apr|jun|jul|aug|"
        r"sep|oct|nov|dec)\.?\s+(20\d{2})", text
    )
    if m:
        return f"{m.group(3)}-{MONTHS.get(m.group(2), '01')}-{m.group(1).zfill(2)}"

    return None


def is_evergreen(url: str) -> bool:
    """Return True if URL belongs to a known live-updating domain."""
    return any(domain in url for domain in EVERGREEN_DOMAINS)


def resolve_date(raw_date, url, content):
    """Full date resolution chain: Tavily → URL → Content → None."""
    return (
        parse_published_date(raw_date)
        or extract_date_from_url(url)
        or extract_date_from_content(content)
    )


# ─────────────────────────────────────────────────────────────
# RECENCY SCORING
# ─────────────────────────────────────────────────────────────
def compute_recency_score(published_date: str | None,
                          url: str = "",
                          days_back: int = DAYS_BACK) -> float:
    """
    Returns recency score 0.0 → 1.0.
      Today         = 1.0
      Older than N  = 0.0  (linear decay)
      Evergreen URL = 0.6  (always current, just undated)
      Unknown date  = 0.3  (neutral)
    """
    if published_date:
        try:
            age   = (datetime.today() - datetime.strptime(published_date, "%Y-%m-%d")).days
            return round(max(0.0, 1.0 - (age / days_back)), 4)
        except Exception:
            pass

    if is_evergreen(url):
        return 0.6    # live-updating page — content is current

    return 0.3        # truly unknown


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def clean_content(text: str, max_len: int = MAX_CONTENT_LENGTH) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\r", " ").strip()
    return text[:max_len] + "..." if len(text) > max_len else text


def deduplicate(articles: list) -> list:
    seen, unique = set(), []
    for a in articles:
        if a["url"] not in seen:
            seen.add(a["url"])
            unique.append(a)
    return unique


# ─────────────────────────────────────────────────────────────
# SEARCH
# ─────────────────────────────────────────────────────────────
def search(query: str, max_results: int = MAX_RESULTS_PER_QUERY) -> list:
    """Run a single Tavily search and return cleaned + dated results."""
    try:
        response = client.search(
            query          = query,
            search_depth   = "advanced",
            max_results    = max_results,
            include_answer = False,
        )
        articles = []
        for r in response.get("results", []):
            url     = r.get("url", "")
            content = clean_content(r.get("content", ""))

            pub_date = resolve_date(
                raw_date = r.get("published_date"),
                url      = url,
                content  = content,
            )

            # Determine date source for reporting
            if parse_published_date(r.get("published_date")):
                date_source = "tavily"
            elif extract_date_from_url(url):
                date_source = "url"
            elif extract_date_from_content(content):
                date_source = "content"
            elif is_evergreen(url):
                date_source = "evergreen"
            else:
                date_source = "none"

            recency = compute_recency_score(pub_date, url)

            articles.append({
                "title"         : r.get("title", "No title"),
                "url"           : url,
                "content"       : content,
                "score"         : round(r.get("score", 0.0), 4),
                "published_date": pub_date,
                "recency_score" : recency,
                "date_source"   : date_source,
            })
        return articles
    except Exception as e:
        print(f"  ⚠️  Search failed for '{query}': {e}")
        return []


# ─────────────────────────────────────────────────────────────
# MAIN FETCHER
# ─────────────────────────────────────────────────────────────
def fetch_news(coin: str) -> dict:
    """Fetch coin-specific + macro news for a given coin."""
    coin = coin.lower()
    if coin not in ["btc", "eth"]:
        raise ValueError("coin must be 'btc' or 'eth'")

    print(f"\n{'='*55}")
    print(f"  TAVILY NEWS FETCHER — {coin.upper()}")
    print(f"  Date : {datetime.today().strftime('%Y-%m-%d')}")
    print(f"{'='*55}")

    print(f"\n  🔍 Searching {coin.upper()} news...")
    coin_articles = []
    for query in QUERIES[coin]:
        results = search(query)
        print(f"    '{query}' → {len(results)} articles")
        coin_articles.extend(results)
    coin_articles = deduplicate(coin_articles)

    print(f"\n  🔍 Searching macro news...")
    macro_articles = []
    for query in MACRO_QUERIES:
        results = search(query)
        print(f"    '{query}' → {len(results)} articles")
        macro_articles.extend(results)
    macro_articles = deduplicate(macro_articles)

    all_articles = deduplicate(coin_articles + macro_articles)
    all_articles = sorted(
        all_articles,
        key=lambda x: (x["recency_score"], x["score"]),
        reverse=True
    )

    # ── Stats
    total  = len(all_articles)
    by_src = {"tavily": 0, "url": 0, "content": 0, "evergreen": 0, "none": 0}
    for a in all_articles:
        by_src[a.get("date_source", "none")] += 1

    print(f"\n  ✅ Total articles fetched : {total}")
    print(f"     Coin-specific          : {len(coin_articles)}")
    print(f"     Macro                  : {len(macro_articles)}")
    print(f"\n  📅 Date resolution:")
    print(f"     Tavily field  : {by_src['tavily']}")
    print(f"     URL extracted : {by_src['url']}")
    print(f"     Content text  : {by_src['content']}")
    print(f"     Evergreen     : {by_src['evergreen']}  (score 0.6 — live pages)")
    print(f"     Not found     : {by_src['none']}  (score 0.3 — neutral)")

    return {
        "coin"          : coin.upper(),
        "date"          : datetime.today().strftime("%Y-%m-%d"),
        "coin_articles" : coin_articles,
        "macro_articles": macro_articles,
        "all_articles"  : all_articles,
        "total"         : total,
    }


def format_for_llm(news: dict, max_articles: int = 8) -> str:
    """Format fetched news into a clean string for LLM prompt injection."""
    lines = [
        f"RECENT NEWS — {news['coin']} ({news['date']})",
        "=" * 50,
    ]
    for i, article in enumerate(news["all_articles"][:max_articles], 1):
        lines.append(f"\n[{i}] {article['title']}")
        lines.append(f"    {article['content']}")
        if article.get("published_date"):
            lines.append(
                f"    Published : {article['published_date']} "
                f"(via {article.get('date_source', '?')})"
            )
        elif article.get("date_source") == "evergreen":
            lines.append(f"    Published : live-updating page")
        lines.append(f"    Source    : {article['url']}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin", type=str, required=True, choices=["btc", "eth"])
    args = parser.parse_args()
    news = fetch_news(args.coin)
    print(f"\n{'='*55}")
    print(format_for_llm(news))