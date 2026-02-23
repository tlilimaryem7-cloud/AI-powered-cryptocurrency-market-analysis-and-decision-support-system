# ============================================================
# LLM — Standalone Debug & Test Script  (v3)
# ============================================================
# Purpose : Run the full ML + news + LLM pipeline from the
#           terminal WITHOUT launching the Streamlit app.
#           Useful for quick debugging and output verification.
#
# Changes vs v2:
#   ✅ FIX 1 — Removed duplicate SYSTEM_PROMPT, build_prompt(),
#              and analyze(). These lived here AND in chatbot.py,
#              meaning debug output didn't match real chatbot output.
#   ✅ FIX 2 — Now imports call_llm() and build_context() directly
#              from chatbot.py → guaranteed identical output to
#              what the live chatbot produces.
#   ✅ FIX 3 — intent parameter added to full_analysis() so you
#              can test any of the 5 intent pipelines, not just
#              "prediction".
# ============================================================

import sys
import os

# ─────────────────────────────────────────────────────────────
# PATH SETUP
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = r"C:\Users\tlili\OneDrive\Bureau\Bootcamp\AI-powered-cryptocurrency-market-analysis-and-decision-support-system"
SRC_PATH     = os.path.join(PROJECT_ROOT, "src")

sys.path.append(PROJECT_ROOT)
sys.path.append(SRC_PATH)


# ─────────────────────────────────────────────────────────────
# FULL PIPELINE — ML + NEWS + LLM
# ─────────────────────────────────────────────────────────────
def full_analysis(question: str, coin: str, intent: str = "prediction") -> dict:
    """
    End-to-end debug pipeline — identical output to the live chatbot.

    Steps:
        1. Run live_pipeline  → ML prediction
        2. Fetch news         → Tavily parallel queries
        3. RAG retrieval      → TF-IDF + keyword boost + recency
        4. Build context      → same function used by chatbot.py
        5. LLM analysis       → same call_llm() used by chatbot.py

    Parameters
    ----------
    question : str — your test question
    coin     : str — "btc" or "eth"
    intent   : str — one of: "prediction", "explanation",
                     "market_overview", "investment_advice",
                     "factual_crypto"
                     Default: "prediction"

    Returns
    -------
    dict with keys:
        question, intent, coin, prediction, articles, context, analysis
    """
    # Import real pipeline components — same ones chatbot.py uses
    from live_pipeline       import predict, build_live_features
    from news.tavily_fetcher import fetch_news
    from news.rag            import retrieve
    from chatbot             import call_llm, build_context

    valid_intents = {
        "prediction", "explanation", "market_overview",
        "investment_advice", "factual_crypto"
    }
    if intent not in valid_intents:
        raise ValueError(f"❌ Unknown intent '{intent}'. Choose from: {valid_intents}")

    coin = coin.lower()
    if coin not in ["btc", "eth"]:
        raise ValueError("❌ coin must be 'btc' or 'eth'")

    print(f"\n{'='*55}")
    print(f"  DEBUG PIPELINE — {coin.upper()} | intent: {intent}")
    print(f"{'='*55}")

    prediction    = None
    live_features = None
    articles      = []

 # ── Step 1: ML prediction (not needed for factual_crypto)
    if intent not in ("factual_crypto", "market_overview"):
        print("\n⚙️  Step 1 : Running ML model...")

        if intent == "explanation":
            # Build features ONCE — reused in both Step 1 and Step 2
            features_series = build_live_features(coin)
            prediction      = predict(coin, features=features_series)
        else:
            # All other intents — normal flow, no Step 2 needed
            features_series = None
            prediction      = predict(coin)

        print(f"  → {prediction['direction']} ({prediction['confidence']}% confidence)")
    else:
        features_series = None
        prediction      = None
        if intent == "factual_crypto":
            print("\n⚙️  Step 1 : Skipped (factual_crypto — no ML needed)")
        else:
            print("\n⚙️  Step 1 : Skipped (market_overview — no ML needed)")

    # ── Step 2: Live features (explanation only)
    if intent == "explanation":
        print("\n⚙️  Step 2 : Using pre-built features from Step 1 — no extra download ✅")
        live_features = features_series.to_dict()
    else:
        print("\n⚙️  Step 2 : Skipped (live features only used for explanation)")
        live_features = None

    # ── Step 3: Fetch + retrieve news (not needed for factual_crypto)
    if intent != "factual_crypto":
        print("\n⚙️  Step 3 : Fetching news...")
        if intent == "market_overview":
            news_btc = fetch_news("btc")
            news_eth = fetch_news("eth")
            all_news = news_btc["all_articles"] + news_eth["all_articles"]
            seen, deduped = set(), []
            for a in all_news:
                if a["url"] not in seen:
                    seen.add(a["url"])
                    deduped.append(a)
            articles = retrieve(question, deduped, "both", top_n=8)
        else:
            news     = fetch_news(coin)
            articles = retrieve(question, news["all_articles"], coin)
        print(f"  → {len(articles)} relevant articles selected")
    else:
        print("\n⚙️  Step 3 : Skipped (factual_crypto — no news needed)")

    # ── Step 4: Build context — same function as chatbot.py
    print("\n⚙️  Step 4 : Building context...")
    context = build_context(
        intent        = intent,
        coin          = coin,
        prediction    = prediction,
        articles      = articles,
        live_features = live_features,
        history       = [],
    )

    # ── Step 5: LLM call — same function as chatbot.py
    print("\n⚙️  Step 5 : Calling LLM...")
    analysis = call_llm(intent, question, context, history=[])

    print("\n✅ Debug analysis complete")

    return {
        "question"  : question,
        "intent"    : intent,
        "coin"      : coin.upper(),
        "prediction": prediction,
        "articles"  : articles,
        "context"   : context,
        "analysis"  : analysis,
    }


# ─────────────────────────────────────────────────────────────
# CLI / QUICK TEST
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Debug the full CoinTrend AI pipeline from the terminal"
    )
    parser.add_argument(
        "--question", type=str,
        default="What will Bitcoin do tomorrow based on current market conditions?",
        help="Question to test"
    )
    parser.add_argument(
        "--coin", type=str, default="btc",
        choices=["btc", "eth"],
        help="Coin to analyse"
    )
    parser.add_argument(
        "--intent", type=str, default="prediction",
        choices=["prediction", "explanation", "market_overview",
                 "investment_advice", "factual_crypto"],
        help="Intent to test"
    )
    args = parser.parse_args()

    result = full_analysis(args.question, args.coin, args.intent)

    print(f"\n{'='*55}")
    print(f"  FINAL ANALYSIS")
    print(f"{'='*55}")
    print(f"  Question : {result['question']}")
    print(f"  Intent   : {result['intent']}")
    print(f"  Coin     : {result['coin']}")
    if result["prediction"]:
        p = result["prediction"]
        print(f"  ML Signal: {p['direction']} ({p['confidence']}% confidence)")
    print(f"\n{'-'*55}")
    print(result["analysis"])
    print(f"{'='*55}")

    # ── Score breakdown for retrieved articles
    if result["articles"]:
        print(f"\n  📰 Articles used ({len(result['articles'])}):")
        for i, a in enumerate(result["articles"], 1):
            b = a.get("score_breakdown", {})
            print(f"  [{i}] {a['title'][:60]}...")
            print(f"       Score: {a['relevance_score']} | "
                  f"tfidf: {b.get('tfidf')} | "
                  f"recency: {b.get('recency')} | "
                  f"keyword: {b.get('keyword')} | "
                  f"tavily: {b.get('tavily')}")