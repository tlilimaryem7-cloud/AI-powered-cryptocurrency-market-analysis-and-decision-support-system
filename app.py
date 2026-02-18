# ============================================================
# COINTREND — "One Step Ahead of the Market"
# ============================================================
# Run : streamlit run app.py
# ============================================================

import sys
import os
import time
import subprocess
import numpy as np

BASE_PATH = r"C:\Users\tlili\OneDrive\Bureau\Bootcamp\AI-powered-cryptocurrency-market-analysis-and-decision-support-system"
sys.path.append(BASE_PATH)
sys.path.append(os.path.join(BASE_PATH, "src"))

import streamlit as st
import pandas    as pd
import yfinance  as yf
import plotly.graph_objects as go
from datetime import datetime

from src.live_pipeline import predict
from chatbot           import chat

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CoinTrend",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────
# COLORS
# ─────────────────────────────────────────────────────────────
NAVY   = "#0f172a"
NAVY2  = "#1e293b"
NAVY3  = "#334155"
BORDER = "#2d3f55"
TEXT   = "#e2e8f0"
MUTED  = "#94a3b8"
BTC    = "#f7931a"
ETH    = "#627eea"
GREEN  = "#10b981"
RED    = "#f43f5e"
INDIGO = "#818cf8"
AMBER  = "#fb923c"

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
FEATURES_PATH = os.path.join(BASE_PATH, "data", "processed", "crypto_features.csv")
PIPELINE_PATH = os.path.join(BASE_PATH, "src", "pipeline.py")
REFRESH_HOURS = 12

# ─────────────────────────────────────────────────────────────
# LOGO
# ─────────────────────────────────────────────────────────────
LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 260 65" width="200" height="50">
  <defs>
    <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#f7931a;stop-opacity:1"/>
      <stop offset="100%" style="stop-color:#627eea;stop-opacity:1"/>
    </linearGradient>
    <linearGradient id="cg" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#f7931a;stop-opacity:1"/>
      <stop offset="100%" style="stop-color:#627eea;stop-opacity:1"/>
    </linearGradient>
  </defs>
  <g transform="translate(8,10)">
    <rect x="0"  y="28" width="6" height="12" rx="1.5" fill="#f7931a" opacity="0.7"/>
    <rect x="9"  y="20" width="6" height="20" rx="1.5" fill="#f7931a" opacity="0.85"/>
    <rect x="18" y="12" width="6" height="28" rx="1.5" fill="url(#cg)"/>
    <rect x="27" y="6"  width="6" height="34" rx="1.5" fill="#627eea" opacity="0.85"/>
    <polyline points="3,34 12,24 21,16 30,8" fill="none"
      stroke="url(#grad)" stroke-width="2"
      stroke-linecap="round" stroke-linejoin="round"/>
    <polygon points="30,4 34,10 26,10" fill="#627eea"/>
  </g>
  <text x="56" y="30" font-family="Arial,sans-serif"
    font-size="22" font-weight="800" fill="url(#grad)"
    letter-spacing="-0.5">CoinTrend</text>
  <line x1="57" y1="38" x2="255" y2="38" stroke="#30363d" stroke-width="0.5"/>
  <text x="57" y="52" font-family="Arial,sans-serif"
    font-size="9" font-weight="400" fill="#8b949e"
    letter-spacing="1.8">ONE STEP AHEAD OF THE MARKET</text>
</svg>"""

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  .stApp {{ background:{NAVY} !important; color:{TEXT}; }}
  .block-container {{ padding:1.2rem 2rem !important; max-width:1400px; }}
  #MainMenu, footer, header {{ visibility:hidden; }}
  .stDeployButton {{ display:none; }}

  /* Cards */
  .card {{
    background:{NAVY2}; border:1px solid {BORDER};
    border-radius:14px; padding:18px 20px; margin:4px 0;
    transition:border-color 0.2s;
  }}
  .card:hover  {{ border-color:{BTC}55; }}
  .card-label  {{ font-size:0.68rem; color:{MUTED}; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px; }}
  .card-value  {{ font-size:1.35rem; font-weight:700; }}
  .card-sub    {{ font-size:0.72rem; color:{MUTED}; margin-top:4px; }}

  /* Section titles */
  .section-title {{
    font-size:0.72rem; font-weight:700; color:{MUTED};
    text-transform:uppercase; letter-spacing:0.12em;
    padding:18px 0 10px 0; border-bottom:1px solid {BORDER}; margin-bottom:14px;
  }}

  /* Hero */
  .hero-card {{
    background:linear-gradient(135deg,{NAVY2} 0%,#0f2540 100%);
    border-radius:20px; padding:32px 28px; border:1px solid {BORDER};
  }}
  .hero-glow-up   {{ box-shadow:0 0 60px {GREEN}22,0 0 120px {GREEN}11; border-color:{GREEN}44 !important; }}
  .hero-glow-down {{ box-shadow:0 0 60px {RED}22,0 0 120px {RED}11;   border-color:{RED}44   !important; }}

  /* Signal pills */
  .signal-pill {{
    display:inline-flex; align-items:center; gap:6px;
    padding:7px 14px; border-radius:999px; font-size:0.78rem; font-weight:600; margin:3px;
  }}
  .pill-bull {{ background:{GREEN}18; border:1px solid {GREEN}55; color:{GREEN}; }}
  .pill-bear {{ background:{RED}18;   border:1px solid {RED}55;   color:{RED}; }}
  .pill-neut {{ background:{MUTED}18; border:1px solid {MUTED}44; color:{MUTED}; }}

  /* Tooltips */
  .tt {{ position:relative; display:inline-block; cursor:help; }}
  .tt .tip {{
    visibility:hidden; opacity:0; background:{NAVY3}; color:{TEXT};
    border:1px solid {BORDER}; border-radius:10px; padding:10px 14px;
    font-size:0.74rem; line-height:1.5; width:230px; position:absolute;
    z-index:9999; bottom:calc(100% + 8px); left:50%; transform:translateX(-50%);
    transition:opacity 0.18s; pointer-events:none; box-shadow:0 8px 32px rgba(0,0,0,0.45);
  }}
  .tt .tip::after {{
    content:""; position:absolute; top:100%; left:50%; transform:translateX(-50%);
    border:6px solid transparent; border-top-color:{BORDER};
  }}
  .tt:hover .tip {{ visibility:visible; opacity:1; }}
  .ii {{
    display:inline-flex; align-items:center; justify-content:center;
    width:15px; height:15px; border-radius:50%; background:{NAVY3}; color:{MUTED};
    font-size:0.62rem; font-weight:700; margin-left:3px; vertical-align:middle;
  }}

  /* Signal matrix */
  .sig-card   {{ background:{NAVY2}; border:1px solid {BORDER}; border-radius:12px; padding:14px 16px; margin:4px 0; }}
  .sig-label  {{ font-size:0.67rem; color:{MUTED}; text-transform:uppercase; letter-spacing:0.09em; margin-bottom:4px; }}
  .sig-value  {{ font-size:1.1rem; font-weight:700; }}
  .sig-interp {{ font-size:0.72rem; margin-top:3px; }}

  /* Streamlit buttons */
  .stButton > button {{
    background:{NAVY2} !important; border:1px solid {BORDER} !important;
    color:{TEXT} !important; border-radius:8px !important;
    font-size:0.82rem !important; transition:all 0.2s !important;
  }}
  .stButton > button:hover {{ border-color:{BTC} !important; color:{BTC} !important; }}

  /* Inputs / selects */
  div[data-testid="stTextInput"] input {{
    background:{NAVY2} !important; border:1px solid {BORDER} !important;
    color:{TEXT} !important; border-radius:10px !important; font-size:0.9rem !important;
  }}
  div[data-testid="stSelectbox"] > div > div {{
    background:{NAVY2} !important; border:1px solid {BORDER} !important;
    color:{TEXT} !important; border-radius:8px !important;
  }}

  /* Chatbot section */
  .chat-wrap {{
    background:{NAVY2}; border:1px solid {BORDER};
    border-radius:20px; padding:24px 28px; margin-top:4px;
  }}
  .chat-history {{
    background:{NAVY}; border:1px solid {BORDER}; border-radius:14px;
    padding:18px; min-height:320px; max-height:420px;
    overflow-y:auto; margin-bottom:18px;
  }}
  .chat-bubble-user {{
    background:linear-gradient(135deg,#1e40af,#2563eb);
    border-radius:16px 16px 4px 16px;
    padding:10px 16px; margin:8px 0 8px auto;
    max-width:76%; width:fit-content;
    font-size:0.875rem; line-height:1.55; color:white; word-break:break-word;
  }}
  .chat-bubble-bot {{
    background:{NAVY3}; border:1px solid {BORDER};
    border-radius:16px 16px 16px 4px;
    padding:12px 16px; margin:8px 0;
    max-width:82%; font-size:0.875rem;
    line-height:1.65; word-break:break-word;
  }}
  .chat-empty-state {{
    display:flex; flex-direction:column; align-items:center;
    justify-content:center; min-height:260px;
    color:{MUTED}; text-align:center;
  }}

  hr {{ border-color:{BORDER} !important; margin:6px 0 !important; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
import re

def md_to_html(text: str) -> str:
    """Convert basic markdown to HTML for chat bubble rendering."""
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*',     r'<em>\1</em>',         text)
    text = re.sub(r'(?m)^\d+\.\s+(.+)$', r'<li>\1</li>', text)
    text = re.sub(r'(<li>.*?</li>)+',    r'<ol>\g<0></ol>', text, flags=re.DOTALL)
    text = re.sub(r'(?m)^[\*\-•]\s+(.+)$', r'<li>\1</li>', text)
    text = re.sub(r'(<li>.*?</li>)+',       r'<ul>\g<0></ul>', text, flags=re.DOTALL)
    text = text.replace('\n', '<br>')
    return text

def hex_rgba(h, a=0.1):
    r, g, b = int(h[1:3],16), int(h[3:5],16), int(h[5:7],16)
    return f"rgba({r},{g},{b},{a})"

def tt(label, tip):
    """Always assign result to a variable before embedding in f-strings."""
    return (f'<span class="tt">{label}'
            f'<span class="ii">i</span>'
            f'<span class="tip">{tip}</span></span>')

def fear_greed_label(s):
    if s <= 24: return "Extreme Fear",  RED
    if s <= 44: return "Fear",          AMBER
    if s <= 55: return "Neutral",       MUTED
    if s <= 74: return "Greed",         "#84cc16"
    return "Extreme Greed",              GREEN

def conf_color(c):
    if c >= 70: return GREEN
    if c >= 58: return AMBER
    return RED

def rsi_state(r):
    if r >= 70: return "Overbought ⚠️", RED,   "bear"
    if r <= 30: return "Oversold 🟢",   GREEN, "bull"
    return f"Neutral ({r:.0f})",          MUTED, "neut"

def macd_state(h):
    if h > 0: return "Bullish ↑", GREEN, "bull"
    return "Bearish ↓",            RED,   "bear"

def bb_state(p):
    if p > 0.8: return "Near Upper — Overbought", RED,   "bear"
    if p < 0.2: return "Near Lower — Oversold",   GREEN, "bull"
    return "Mid Band — Neutral",                   MUTED, "neut"

def mom_state(a):
    if a > 0.01:  return "Accelerating Up ↑", GREEN, "bull"
    if a < -0.01: return "Decelerating ↓",    RED,   "bear"
    return "Flat →",                            MUTED, "neut"

def vix_state(v):
    if v > 30: return "Extreme Fear ⚠️", RED,   "bear"
    if v > 20: return "Elevated",         AMBER, "neut"
    return "Low — Risk On ✅",             GREEN, "bull"

def dxy_state(d):
    if d > 0.001:  return "Strong ↑ — Bearish Crypto", RED,   "bear"
    if d > 0:      return "Mild ↑ — Slight Headwind",   AMBER, "neut"
    if d > -0.001: return "Mild ↓ — Slight Tailwind",   GREEN, "bull"
    return "Weak ↓ — Bullish Crypto",                    GREEN, "bull"

def fg_bar(score):
    lb, col = fear_greed_label(score)
    return f"""
    <div style="margin:8px 0 2px 0;">
      <div style="position:relative;height:10px;border-radius:5px;
          background:linear-gradient(to right,{RED} 0%,{AMBER} 25%,#eab308 50%,#84cc16 75%,{GREEN} 100%);">
        <div style="position:absolute;left:{score}%;transform:translateX(-50%);top:-5px;
            width:4px;height:20px;background:white;border-radius:2px;
            box-shadow:0 0 8px rgba(255,255,255,0.9);"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:0.62rem;color:{MUTED};margin-top:6px;">
        <span>Extreme Fear</span>
        <span style="color:{col};font-weight:700;">{int(score)} — {lb}</span>
        <span>Extreme Greed</span>
      </div>
    </div>"""

def gauge_bar(pct, color):
    pct = max(0, min(100, pct))
    return (f'<div style="background:{NAVY3};border-radius:4px;height:7px;margin:6px 0 3px 0;">'
            f'<div style="background:{color};width:{pct:.0f}%;height:7px;border-radius:4px;"></div>'
            f'</div>')

def chart_base(fig, h=280):
    fig.update_layout(
        paper_bgcolor=NAVY2, plot_bgcolor=NAVY2,
        margin=dict(l=10,r=10,t=14,b=10), height=h,
        showlegend=False, font=dict(color=MUTED, size=10), hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, color=MUTED, tickfont=dict(size=9),
                     zeroline=False, showspikes=True, spikecolor=MUTED,
                     spikethickness=1, spikedash="dot")
    fig.update_yaxes(showgrid=True, gridcolor=NAVY3, color=MUTED,
                     tickfont=dict(size=9), zeroline=False)
    return fig

def period_dd(key, default="90D"):
    opts = {"7D":7,"30D":30,"90D":90,"180D":180,"1Y":365}
    sel  = st.selectbox("", list(opts.keys()),
                        index=list(opts.keys()).index(default),
                        key=f"dd_{key}", label_visibility="collapsed")
    return opts[sel]


# ─────────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_df():
    rows = []
    for coin, ticker in [("btc","BTC-USD"),("eth","ETH-USD")]:
        raw = yf.download(ticker, period="1y", interval="1d", progress=False)
        raw = raw.reset_index()
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
        p = raw["Close"].squeeze()

        delta  = p.diff()
        gain   = delta.clip(lower=0).rolling(14).mean()
        loss   = (-delta.clip(upper=0)).rolling(14).mean()
        rsi    = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
        ema12  = p.ewm(span=12, adjust=False).mean()
        ema26  = p.ewm(span=26, adjust=False).mean()
        macd_l = ema12 - ema26
        macd_s = macd_l.ewm(span=9, adjust=False).mean()
        macd_h = macd_l - macd_s
        ma20   = p.rolling(20).mean()
        std20  = p.rolling(20).std()
        bb     = (p - (ma20 - 2*std20)) / ((4*std20).replace(0, np.nan))
        ret    = p.pct_change()
        vol7   = ret.rolling(7).std()
        ma7    = p.rolling(7).mean()
        ma30   = p.rolling(30).mean()
        ma50   = p.rolling(50).mean()
        mom_acc = p.pct_change(5) - p.pct_change(10)

        rows.append(pd.DataFrame({
            "timestamp"            : raw["Date"],
            "coin"                 : coin,
            "price"                : p.values,
            "volume"               : raw["Volume"].squeeze().values,
            "rsi_14"               : rsi.values,
            "macd"                 : macd_l.values,
            "macd_signal"          : macd_s.values,
            "macd_histogram"       : macd_h.values,
            "bb_pct"               : bb.values,
            "volatility_7d"        : vol7.values,
            "ma_7"                 : ma7.values,
            "ma_30"                : ma30.values,
            "ma_50"                : ma50.values,
            "momentum_acceleration": mom_acc.values,
            "bull_bear_flag"       : (p > ma50).astype(int).values,
        }))
    return pd.concat(rows, ignore_index=True)

@st.cache_data(ttl=3600, show_spinner=False)
def load_macro():
    return pd.read_csv(FEATURES_PATH, parse_dates=["timestamp"])

@st.cache_data(ttl=1800, show_spinner=False)
def get_pred(coin): return predict(coin)


# ─────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────
def chart_price(df, coin, days, color):
    sub = df[df["coin"]==coin].tail(days)
    fig = go.Figure()
    for col, c, dash, w in [("ma_50",MUTED,"dot",1.0),("ma_30",AMBER,"dash",1.1),("ma_7",INDIGO,"solid",1.2)]:
        if col in sub.columns:
            fig.add_trace(go.Scatter(x=sub["timestamp"], y=sub[col], mode="lines",
                name=col.upper(), line=dict(color=c,width=w,dash=dash),
                hovertemplate=f"{col}: $%{{y:,.0f}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=sub["timestamp"], y=sub["price"], mode="lines",
        name="Price", line=dict(color=color,width=2.2),
        fill="tozeroy", fillcolor=hex_rgba(color,0.07),
        hovertemplate="Price: $%{y:,.2f}<extra></extra>"))
    fig = chart_base(fig, h=320)
    fig.update_yaxes(tickformat="$,.0f")
    fig.update_layout(showlegend=True,
        legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="right",x=1,
                    font=dict(size=9),bgcolor="rgba(0,0,0,0)"))
    return fig

def chart_rsi(df, coin, days):
    sub = df[df["coin"]==coin].tail(days)
    fig = go.Figure()
    fig.add_hrect(y0=70,y1=100,fillcolor=hex_rgba(RED,0.06),line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor=hex_rgba(GREEN,0.06),line_width=0)
    fig.add_trace(go.Scatter(x=sub["timestamp"],y=sub["rsi_14"],mode="lines",
        line=dict(color=INDIGO,width=2),hovertemplate="RSI: %{y:.1f}<extra></extra>"))
    fig.add_hline(y=70,line_dash="dash",line_color=RED,  opacity=0.5)
    fig.add_hline(y=30,line_dash="dash",line_color=GREEN,opacity=0.5)
    fig.add_hline(y=50,line_dash="dot", line_color=MUTED,opacity=0.3)
    fig = chart_base(fig, h=220)
    fig.update_yaxes(range=[0,100])
    fig.update_layout(annotations=[
        dict(x=1,y=75,xref="paper",yref="y",text="Overbought",font=dict(size=9,color=RED),  showarrow=False,xanchor="right"),
        dict(x=1,y=25,xref="paper",yref="y",text="Oversold",  font=dict(size=9,color=GREEN),showarrow=False,xanchor="right"),
    ])
    return fig

def chart_macd(df, coin, days):
    sub    = df[df["coin"]==coin].tail(days)
    colors = [GREEN if v >= 0 else RED for v in sub["macd_histogram"]]
    fig    = go.Figure()
    fig.add_trace(go.Bar(x=sub["timestamp"],y=sub["macd_histogram"],
        marker_color=colors,hovertemplate="Hist: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=sub["timestamp"],y=sub["macd"],mode="lines",
        line=dict(color=BTC,width=1.4),name="MACD",hovertemplate="MACD: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=sub["timestamp"],y=sub["macd_signal"],mode="lines",
        line=dict(color=ETH,width=1.4,dash="dash"),name="Signal",hovertemplate="Signal: %{y:.4f}<extra></extra>"))
    fig.add_hline(y=0,line_color=MUTED,line_width=0.8,opacity=0.5)
    fig = chart_base(fig, h=240)
    fig.update_layout(showlegend=True,
        legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="right",x=1,
                    font=dict(size=9),bgcolor="rgba(0,0,0,0)"))
    return fig


# ─────────────────────────────────────────────────────────────
# SIGNALS
# ─────────────────────────────────────────────────────────────
def build_signals(feat_live, feat_macro):
    signals = []
    rsi  = float(feat_live["rsi_14"])
    macd = float(feat_live["macd_histogram"])
    mom  = float(feat_live["momentum_acceleration"])
    bull = int(feat_live["bull_bear_flag"]) == 1
    try:
        vix = float(feat_macro["vix"])
        dxy = float(feat_macro["dxy_return_ma7"])
        fg  = float(feat_macro["fear_greed"])
    except Exception:
        vix, dxy, fg = 20.0, 0.0, 50.0

    if rsi <= 35:     signals.append(("RSI rising from oversold zone",           "bull"))
    elif rsi >= 65:   signals.append(("RSI approaching overbought territory",    "bear"))
    else:             signals.append((f"RSI neutral at {rsi:.0f}",               "neut"))
    if macd > 0:      signals.append(("MACD histogram turning positive",         "bull"))
    else:             signals.append(("MACD histogram in negative territory",    "bear"))
    if mom > 0.01:    signals.append(("Short-term momentum accelerating upward", "bull"))
    elif mom < -0.01: signals.append(("Short-term momentum decelerating",        "bear"))
    if vix < 18:      signals.append(("Low VIX — risk-on environment",           "bull"))
    elif vix > 28:    signals.append(("High VIX — elevated market fear",         "bear"))
    if dxy < -0.001:  signals.append(("Dollar weakening — bullish tailwind",     "bull"))
    elif dxy > 0.001: signals.append(("Dollar strengthening — headwind",         "bear"))
    if fg <= 25:      signals.append(("Extreme Fear — contrarian opportunity",   "neut"))
    elif fg >= 70:    signals.append(("Extreme Greed — caution advised",         "bear"))
    if bull:          signals.append(("Price above MA50 — bull market regime",   "bull"))
    else:             signals.append(("Price below MA50 — bear market regime",   "bear"))
    return signals[:6]


# ─────────────────────────────────────────────────────────────
# SMART REFRESH
# ─────────────────────────────────────────────────────────────
def smart_refresh():
    if os.path.exists(FEATURES_PATH):
        if (time.time() - os.path.getmtime(FEATURES_PATH)) / 3600 <= REFRESH_HOURS:
            return
    with st.spinner("🔄 Updating market data..."):
        try:
            subprocess.run(["python", PIPELINE_PATH],
                           capture_output=True, text=True, timeout=120)
        except Exception as e:
            st.warning(f"⚠️ Data update failed: {e}")


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
for k, v in {"coin":"btc","chat_history":[],"preds":{}}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────
# BOOT
# ─────────────────────────────────────────────────────────────
smart_refresh()
df_live  = load_df()
df_macro = load_macro()

with st.spinner("⚙️ Loading AI predictions..."):
    for c in ["btc","eth"]:
        if c not in st.session_state.preds:
            st.session_state.preds[c] = get_pred(c)

coin     = st.session_state.coin
pred     = st.session_state.preds[coin]
color    = BTC if coin == "btc" else ETH
name     = "Bitcoin" if coin == "btc" else "Ethereum"
is_up    = pred["direction"].startswith("UP")
dir_col  = GREEN if is_up else RED
glow_cls = "hero-glow-up" if is_up else "hero-glow-down"

feat_live  = df_live[df_live["coin"]==coin].dropna(subset=["rsi_14"]).iloc[-1]
feat_macro = df_macro[df_macro["coin"]==coin].iloc[-1]
signals    = build_signals(feat_live, feat_macro)

last_upd = datetime.fromtimestamp(
    os.path.getmtime(FEATURES_PATH)
).strftime("%b %d %H:%M")


# ─────────────────────────────────────────────────────────────
# PRE-COMPUTE ALL TOOLTIPS  (never call tt() inside f-strings)
# ─────────────────────────────────────────────────────────────
tt_conviction = tt("AI Conviction",
    "How strongly the model signals align. "
    "Above 70% = strong. 55–70% = moderate. Below 55% = uncertain.")
tt_fg  = tt("Fear &amp; Greed Index",
    "Sentiment 0–100. Extreme Fear = potential buy. Extreme Greed = caution.")
tt_vix = tt("VIX — Market Fear",
    "Stock market volatility. Above 25 = risk-off. Below 15 = risk-on.")
tt_dxy = tt("DXY — Dollar Trend",
    "Rising DXY = bearish for crypto. Falling DXY = bullish.")
tt_rsi  = tt("RSI 14",
    "Above 70 = overbought. Below 30 = oversold. 50 = neutral.")
tt_macd = tt("MACD Histogram",
    "Positive & growing = bullish momentum. Negative & falling = bearish.")
tt_bb   = tt("Bollinger Position",
    "0 = lower band (oversold). 1 = upper band (overbought).")
tt_mom  = tt("Momentum",
    "Positive: price accelerating. Negative: momentum slowing.")
tt_vol  = tt("Volatility 7d",
    "High = larger swings. Low = calmer market.")
tt_reg  = tt("Market Regime",
    "Bull: price above MA50. Bear: price below MA50.")


# ═════════════════════════════════════════════════════════════
# NAVBAR
# ═════════════════════════════════════════════════════════════
n1, n2, n3 = st.columns([3,3,4])
with n1:
    st.markdown(LOGO_SVG, unsafe_allow_html=True)
with n2:
    ca, cb = st.columns(2)
    with ca:
        if st.button("₿  Bitcoin",  key="nav_btc",
                     type="primary" if coin=="btc" else "secondary",
                     use_container_width=True):
            st.session_state.coin = "btc"; st.rerun()
    with cb:
        if st.button("Ξ  Ethereum", key="nav_eth",
                     type="primary" if coin=="eth" else "secondary",
                     use_container_width=True):
            st.session_state.coin = "eth"; st.rerun()
with n3:
    price_now = float(feat_live["price"])
    st.markdown(
        f'<div style="text-align:right;padding-top:10px;">'
        f'<span style="font-size:0.75rem;color:{MUTED};">'
        f'📅 {datetime.now().strftime("%b %d, %Y")}'
        f'&nbsp;|&nbsp; Updated: {last_upd}'
        f'&nbsp;|&nbsp; Price: <strong style="color:{color};">${price_now:,.2f}</strong>'
        f'</span></div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# 🎯 HERO — AI VERDICT
# ═════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🎯 AI Verdict</div>', unsafe_allow_html=True)

cc   = conf_color(pred["confidence"])
conf = pred["confidence"]
dir_text = pred["direction"]

h_left, h_right = st.columns([1,1])

with h_left:
    st.markdown(
        f'<div class="hero-card {glow_cls}">'
        f'<div style="font-size:0.72rem;color:{MUTED};text-transform:uppercase;'
        f'letter-spacing:0.12em;margin-bottom:14px;">{name} — Tomorrow\'s Direction</div>'
        f'<div style="display:flex;align-items:center;justify-content:space-between;gap:16px;">'
        f'<div style="font-size:3rem;font-weight:900;color:{dir_col};line-height:1;flex-shrink:0;">{dir_text}</div>'
        f'<div style="flex:1;min-width:0;">'
        f'<div style="margin-bottom:4px;">{tt_conviction}</div>'
        f'<div style="font-size:1.8rem;font-weight:800;color:{cc};">{conf}%</div>'
        f'<div style="background:{NAVY3};border-radius:6px;height:8px;margin-top:8px;">'
        f'<div style="background:linear-gradient(90deg,{NAVY3},{cc});width:{conf}%;height:8px;border-radius:6px;"></div>'
        f'</div></div></div>'
        f'<div style="font-size:0.70rem;color:{MUTED};margin-top:16px;font-style:italic;">'
        f'⚠️ Not a financial advice — for decision support only</div>'
        f'</div>', unsafe_allow_html=True)

with h_right:
    try:
        fg  = float(feat_macro["fear_greed"])
        vix = float(feat_macro["vix"])
        dxy = float(feat_macro["dxy_return_ma7"])
    except Exception:
        fg, vix, dxy = 50.0, 20.0, 0.0

    vs, vc_ = vix_state(vix)[:2]
    ds, dc_ = dxy_state(dxy)[:2]
    dxy_dir = "↑ Strong" if dxy > 0 else "↓ Weak"
    fg_html = fg_bar(fg)

    st.markdown(
        f'<div class="card" style="margin-bottom:10px;">'
        f'<div class="card-label">{tt_fg}</div>{fg_html}</div>',
        unsafe_allow_html=True)
    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown(
            f'<div class="card"><div class="card-label">{tt_vix}</div>'
            f'<div class="card-value" style="color:{vc_};">{vix:.1f}</div>'
            f'<div class="card-sub" style="color:{vc_};">{vs}</div></div>',
            unsafe_allow_html=True)
    with mc2:
        st.markdown(
            f'<div class="card"><div class="card-label">{tt_dxy}</div>'
            f'<div class="card-value" style="color:{dc_};">{dxy_dir}</div>'
            f'<div class="card-sub" style="color:{dc_};">{ds}</div></div>',
            unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# 🔑 TOP SIGNALS
# ═════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🔑 Top Signals Driving This Prediction</div>',
            unsafe_allow_html=True)
pills = "".join(
    f'<span class="signal-pill pill-{d}">{"▲" if d=="bull" else "▼" if d=="bear" else "→"} {t}</span>'
    for t, d in signals
)
st.markdown(f'<div style="padding:4px 0 12px 0;">{pills}</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# 📊 SIGNAL MATRIX
# ═════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">📊 Signal Matrix</div>', unsafe_allow_html=True)

rsi_v  = float(feat_live["rsi_14"])
macd_v = float(feat_live["macd_histogram"])
bb_v   = float(feat_live["bb_pct"])
mom_v  = float(feat_live["momentum_acceleration"])
vol7_v = float(feat_live["volatility_7d"]) * 100
bull_v = int(feat_live["bull_bear_flag"]) == 1

ri,  rc,  _ = rsi_state(rsi_v)
mi,  mc_, _ = macd_state(macd_v)
bi,  bc,  _ = bb_state(bb_v)
mo,  moc, _ = mom_state(mom_v)
vol_c = RED if vol7_v > 3 else GREEN
reg_c = GREEN if bull_v else RED

sc1,sc2,sc3 = st.columns(3)
sc4,sc5,sc6 = st.columns(3)

matrix = [
    (sc1, tt_rsi,  f"{rsi_v:.1f}",   ri,                                   rc,   rsi_v),
    (sc2, tt_macd, f"{macd_v:.4f}",  mi,                                   mc_,  50+(50 if macd_v>0 else -50)),
    (sc3, tt_bb,   f"{bb_v:.2f}",    bi,                                   bc,   bb_v*100),
    (sc4, tt_mom,  f"{mom_v:.4f}",   mo,                                   moc,  50+mom_v*1000),
    (sc5, tt_vol,  f"{vol7_v:.2f}%", "High ⚠️" if vol7_v>3 else "Low ✅", vol_c, min(vol7_v*10,100)),
    (sc6, tt_reg,  "Bull 🐂" if bull_v else "Bear 🐻",
                    "Above MA50" if bull_v else "Below MA50",               reg_c, 100 if bull_v else 0),
]
for col, lbl, val, interp, icol, gpct in matrix:
    with col:
        st.markdown(
            f'<div class="sig-card"><div class="sig-label">{lbl}</div>'
            f'<div class="sig-value" style="color:{icol};">{val}</div>'
            f'{gauge_bar(gpct,icol)}'
            f'<div class="sig-interp" style="color:{icol};">{interp}</div></div>',
            unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# 📈 DEEP TECHNICAL VIEW
# ═════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">📈 Deep Technical View</div>', unsafe_allow_html=True)

t1, t2 = st.columns([5,1])
with t1:
    st.markdown(
        f'<span style="font-weight:600;">Price &amp; Moving Averages</span> '
        f'<span style="font-size:0.72rem;color:{MUTED};">'
        f'MA7 <span style="color:{INDIGO};">━</span> &nbsp;'
        f'MA30 <span style="color:{AMBER};">╌</span> &nbsp;'
        f'MA50 <span style="color:{MUTED};">···</span></span>',
        unsafe_allow_html=True)
with t2:
    d1 = period_dd("price","90D")
st.plotly_chart(chart_price(df_live,coin,d1,color),
                use_container_width=True, config={"displayModeBar":False})

rc1, rc2 = st.columns(2)
with rc1:
    r1, r2 = st.columns([4,1])
    with r1:
        st.markdown(
            f'<span style="font-weight:600;">RSI 14</span> '
            f'<span style="font-size:0.72rem;color:{MUTED};">'
            f'Overbought &gt;70 · Oversold &lt;30</span>',
            unsafe_allow_html=True)
    with r2:
        d2 = period_dd("rsi","90D")
    st.plotly_chart(chart_rsi(df_live,coin,d2),
                    use_container_width=True, config={"displayModeBar":False})
with rc2:
    r3, r4 = st.columns([4,1])
    with r3:
        st.markdown(
            f'<span style="font-weight:600;">MACD</span> '
            f'<span style="font-size:0.72rem;color:{MUTED};">'
            f'MACD <span style="color:{BTC};">━</span> &nbsp;'
            f'Signal <span style="color:{ETH};">╌</span></span>',
            unsafe_allow_html=True)
    with r4:
        d3 = period_dd("macd","90D")
    st.plotly_chart(chart_macd(df_live,coin,d3),
                    use_container_width=True, config={"displayModeBar":False})


# ═════════════════════════════════════════════════════════════
# 💬 AI ANALYST — CLASSIC CHATBOT SECTION
# Simple, reliable, fully in Streamlit's normal flow:
#   • Message history rendered as HTML bubbles
#   • 6 quick-question buttons in 2 rows of 3
#   • Text input + Send button + Clear button
# ═════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">💬 AI Analyst</div>', unsafe_allow_html=True)

QUICK_QS = [
    ("₿ BTC tomorrow?",    "What will Bitcoin do tomorrow?"),
    ("Ξ ETH tomorrow?",    "What will Ethereum do tomorrow?"),
    ("⚠️ Main risks?",      f"What are the main risks for {name} today?"),
    ("🌍 Market overview?", "Give me a full crypto market overview"),
    ("📉 Why falling?",  f"Why is {name} falling today?") if is_up else ("📈 Why rising?", f"Why is {name} rising today?"),
    ("📈 Why rising?",   f"Why is {name} rising today?") if is_up else ("📉 Why falling?", f"Why is {name} falling today?"),
]

with st.container():
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

    # ── 1. Message history ──
    if st.session_state.chat_history:
        bubbles = "".join(
            f'<div class="chat-bubble-user">{m["content"]}</div>'
            if m["role"] == "user"
            else f'<div class="chat-bubble-bot">{md_to_html(m["content"])}</div>'
            for m in st.session_state.chat_history
        )
        st.markdown(f'<div class="chat-history">{bubbles}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="chat-history">'
            f'<div class="chat-empty-state">'
            f'<div style="font-size:2.2rem;margin-bottom:10px;">✦</div>'
            f'<div style="font-weight:600;font-size:1rem;color:{TEXT};margin-bottom:6px;">'
            f'CoinTrend AI Analyst</div>'
            f'<div>Ask me anything about {name} or the crypto market.</div>'
            f'</div></div>',
            unsafe_allow_html=True)
        
        # ── 2. Text input + Send + Clear ──
    inp_col, send_col, clear_col = st.columns([7, 1, 1])

    with inp_col:
        user_msg = st.text_input(
            "chat_input_label",
            placeholder="Ask about BTC, ETH, market conditions…",
            key="chat_input",
            label_visibility="collapsed"
        )
    with send_col:
        send_clicked = st.button("Send ➤", key="chat_send", use_container_width=True)
    with clear_col:
        if st.button("🗑️ Clear", key="chat_clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # ── 3. Quick questions ──
    st.markdown(
        f'<div style="font-size:0.68rem;color:{MUTED};text-transform:uppercase;'
        f'letter-spacing:0.09em;margin:4px 0 8px 0;">Quick questions</div>',
        unsafe_allow_html=True)

    with st.container():
        q_row1 = st.columns([1,1,1], gap="small")
        q_row2 = st.columns([1,1,1], gap="small")
        all_q_cols = list(q_row1) + list(q_row2)
        for i, (lbl, question) in enumerate(QUICK_QS):
            with all_q_cols[i]:
                if st.button(lbl, key=f"qq_{i}", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": question})
                    with st.spinner("Analyzing…"):
                        result = chat(question)
                    st.session_state.chat_history.append({"role": "assistant", "content": result["analysis"]})
                    st.rerun()

    # Handle send
    if send_clicked and user_msg and user_msg.strip():
        st.session_state.chat_history.append(
            {"role": "user", "content": user_msg.strip()})
        with st.spinner("Analyzing…"):
            result = chat(user_msg.strip())
        st.session_state.chat_history.append(
            {"role": "assistant", "content": result["analysis"]})
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)  # close .chat-wrap


# ═════════════════════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════════════════════
st.markdown(
    f'<div style="text-align:center;color:{MUTED};font-size:0.72rem;'
    f'margin-top:50px;padding:20px;border-top:1px solid {BORDER};">'
    f'⚠️ CoinTrend is an AI-powered decision support tool. '
    f'This is <strong>not a financial advice</strong>. '
    f'Always do your own research before investing.<br><br>'
    f'<span style="background:linear-gradient(90deg,{BTC},{ETH});'
    f'-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
    f'font-weight:700;">CoinTrend</span> © 2026 — One Step Ahead of the Market'
    f'</div>',
    unsafe_allow_html=True
)