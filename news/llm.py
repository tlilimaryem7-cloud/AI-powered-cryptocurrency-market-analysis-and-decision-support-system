# ============================================================
# LLM — Groq Integration for Crypto Market Analysis
# ============================================================
# Input  : ML prediction + RAG context (news articles)
# Output : Natural language market analysis and recommendation
#
# Model  : llama-3.3-70b-versatile (free via Groq)
# ============================================================

import os
from dotenv import load_dotenv
from groq   import Groq

# ─────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────
ENV_PATH = r"C:\Users\tlili\OneDrive\Bureau\Bootcamp\AI-powered-cryptocurrency-market-analysis-and-decision-support-system\news\.env"

load_dotenv(ENV_PATH)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in .env file")

client = Groq(api_key=GROQ_API_KEY)

MODEL      = "llama-3.3-70b-versatile"
MAX_TOKENS = 1024
TEMPERATURE = 0.3   # low = more factual, less creative


# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a professional cryptocurrency market analyst assistant.

Your role is to:
1. Analyze ML model predictions for BTC and ETH price direction
2. Contextualize the prediction with recent market news
3. Identify supporting and contradicting signals
4. Provide a clear, structured market analysis

Your analysis must always follow this structure:
- 📊 ML Signal: summarize the model prediction and confidence
- 📰 News Context: summarize the most relevant news findings
- ✅ Supporting Factors: what aligns with the prediction
- ⚠️  Risk Factors: what could contradict or invalidate the prediction
- 💡 Final Assessment: your overall conclusion (1-2 sentences)

Important rules:
- Always be factual and data-driven
- Never guarantee profits or promise returns
- Always include a disclaimer that this is not financial advice
- Be concise — total response should be under 300 words
- Use the exact confidence percentage from the ML model
"""


# ─────────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────────
def build_prompt(question: str, context: str) -> str:
    """
    Build the user prompt combining the question and RAG context.

    Parameters
    ----------
    question : str — user's original question
    context  : str — output of rag.format_context()

    Returns
    -------
    str — full prompt for the LLM
    """
    return f"""User Question: {question}

Here is the data context for your analysis:

{context}

Please provide a structured market analysis following the format defined in your instructions.
Base your analysis strictly on the ML prediction and news context provided above.
"""


# ─────────────────────────────────────────────────────────────
# MAIN LLM FUNCTION
# ─────────────────────────────────────────────────────────────
def analyze(question: str, context: str) -> str:
    """
    Generate a market analysis using Groq LLM.

    Parameters
    ----------
    question : str — user's original question
    context  : str — formatted context from rag.format_context()

    Returns
    -------
    str — LLM generated analysis
    """
    prompt = build_prompt(question, context)

    try:
        response = client.chat.completions.create(
            model    = MODEL,
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens  = MAX_TOKENS,
            temperature = TEMPERATURE,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"❌ LLM analysis failed: {e}"


# ─────────────────────────────────────────────────────────────
# FULL PIPELINE — ML + NEWS + LLM
# ─────────────────────────────────────────────────────────────
def full_analysis(question: str, coin: str) -> dict:
    """
    End-to-end pipeline:
    1. Run live_pipeline → ML prediction
    2. Fetch news via Tavily
    3. Retrieve relevant articles via RAG
    4. Generate LLM analysis

    Parameters
    ----------
    question : str — user's question
    coin     : str — "btc" or "eth"

    Returns
    -------
    dict with keys:
        question, coin, prediction, articles, context, analysis
    """
    import sys
    sys.path.append(
        r"C:\Users\tlili\OneDrive\Bureau\Bootcamp"
        r"\AI-powered-cryptocurrency-market-analysis-and-decision-support-system\src"
    )
    sys.path.append(
        r"C:\Users\tlili\OneDrive\Bureau\Bootcamp"
        r"\AI-powered-cryptocurrency-market-analysis-and-decision-support-system"
    )

    from live_pipeline      import predict
    from news.tavily_fetcher import fetch_news
    from news.rag            import retrieve, format_context

    print(f"\n{'='*55}")
    print(f"  FULL ANALYSIS PIPELINE — {coin.upper()}")
    print(f"{'='*55}")

    # Step 1 — ML prediction
    print("\n⚙️  Step 1 : Running ML model...")
    prediction = predict(coin)

    # Step 2 — Fetch news
    print("\n⚙️  Step 2 : Fetching news...")
    news = fetch_news(coin)

    # Step 3 — RAG retrieval
    print("\n⚙️  Step 3 : Retrieving relevant articles...")
    top_articles = retrieve(question, news["all_articles"], coin)
    print(f"  Selected {len(top_articles)} relevant articles")

    # Step 4 — Format context
    context = format_context(top_articles, coin, prediction)

    # Step 5 — LLM analysis
    print("\n⚙️  Step 4 : Generating LLM analysis...")
    analysis = analyze(question, context)

    print("\n✅ Analysis complete")

    return {
        "question"  : question,
        "coin"      : coin.upper(),
        "prediction": prediction,
        "articles"  : top_articles,
        "context"   : context,
        "analysis"  : analysis,
    }


# ─────────────────────────────────────────────────────────────
# CLI / QUICK TEST
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    question = "What will Bitcoin do tomorrow based on current market conditions?"
    result   = full_analysis(question, "btc")

    print(f"\n{'='*55}")
    print(f"  FINAL ANALYSIS")
    print(f"{'='*55}")
    print(result["analysis"])