# ============================================================
# COINTREND — "One step ahead of the market"
# ============================================================
# Multi-page Streamlit App
# Pages : Home | BTC Detail | ETH Detail
# Chat  : Floating chatbot on all pages
#
# Run   : streamlit run app.py
# ============================================================

import sys
import os
import time
import subprocess

BASE_PATH = r"C:\Users\tlili\OneDrive\Bureau\Bootcamp\AI-powered-cryptocurrency-market-analysis-and-decision-support-system"
sys.path.append(BASE_PATH)
sys.path.append(os.path.join(BASE_PATH, "src"))

import streamlit as st
import pandas    as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime        import datetime

from src.live_pipeline   import predict
from news.tavily_fetcher import fetch_news
from news.rag            import retrieve, format_context
from news.llm            import analyze

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "CoinTrend",
    page_icon  = "📈",
    layout     = "wide",
    initial_sidebar_state = "collapsed"
)

# ─────────────────────────────────────────────────────────────
# THEME COLORS
# ─────────────────────────────────────────────────────────────
NAVY   = "#0f172a"
NAVY2  = "#1e293b"
NAVY3  = "#334155"
BORDER = "#334155"
TEXT   = "#e2e8f0"
MUTED  = "#94a3b8"
BTC    = "#f7931a"
ETH    = "#627eea"
GREEN  = "#22c55e"
RED    = "#ef4444"
PURPLE = "#a855f7"
AMBER  = "#f59e0b"

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
FEATURES_PATH = os.path.join(BASE_PATH, "data", "processed", "crypto_features.csv")
PIPELINE_PATH = os.path.join(BASE_PATH, "src", "pipeline.py")
LOGO_PATH     = os.path.join(BASE_PATH, "assets", "cointrend_logo.svg")
REFRESH_HOURS = 12

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    .stApp {{ background-color:{NAVY} !important; color:{TEXT}; }}
    .block-container {{ padding:1.5rem 2rem !important; }}
    #MainMenu, footer, header {{ visibility:hidden; }}
    .stDeployButton {{ display:none; }}

    .card {{
        background:{NAVY2}; border:1px solid {BORDER};
        border-radius:12px; padding:16px 20px; margin:4px 0;
    }}
    .card-label {{
        font-size:0.70rem; color:{MUTED}; text-transform:uppercase;
        letter-spacing:0.08em; margin-bottom:4px;
    }}
    .card-value {{ font-size:1.4rem; font-weight:700; }}
    .card-sub   {{ font-size:0.73rem; color:{MUTED}; margin-top:3px; }}

    .section-title {{
        font-size:0.75rem; font-weight:600; color:{MUTED};
        text-transform:uppercase; letter-spacing:0.1em;
        padding:16px 0 8px 0; border-bottom:1px solid {BORDER};
        margin-bottom:12px;
    }}
    .pred-card {{
        background:{NAVY2}; border-radius:16px; padding:28px 24px;
        border:1px solid {BORDER}; text-align:center;
    }}
    .stButton > button {{
        background:{NAVY2} !important; border:1px solid {BORDER} !important;
        color:{TEXT} !important; border-radius:8px !important;
        font-size:0.82rem !important;
    }}
    .stButton > button:hover {{
        border-color:{BTC} !important;
    }}
    div[data-testid="stTextInput"] input {{
        background:{NAVY2} !important; border:1px solid {BORDER} !important;
        color:{TEXT} !important; border-radius:8px !important;
    }}
    hr {{ border-color:{BORDER} !important; }}
      
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def fear_greed_label(s):
    if s <= 24: return "Extreme Fear",  RED
    if s <= 44: return "Fear",          "#f97316"
    if s <= 55: return "Neutral",       AMBER
    if s <= 74: return "Greed",         "#84cc16"
    return "Extreme Greed",              GREEN

def conf_color(c):
    if c >= 70: return GREEN
    if c >= 60: return AMBER
    return RED

def rsi_label(r):
    if r >= 70: return "Overbought", RED
    if r <= 30: return "Oversold",   GREEN
    return "Neutral",                 MUTED

def macd_label(h):
    return ("Bullish ↑", GREEN) if h > 0 else ("Bearish ↓", RED)

def bb_label(p):
    if p > 0.8: return "Near Upper — Overbought", RED
    if p < 0.2: return "Near Lower — Oversold",   GREEN
    return "Mid Band — Neutral",                   MUTED

def mom_label(a):
    if a > 0.01:  return "Accelerating Up ↑",   GREEN
    if a < -0.01: return "Accelerating Down ↓",  RED
    return "Neutral →",                           MUTED

def hex_rgba(h, a=0.1):
    r,g,b = int(h[1:3],16), int(h[3:5],16), int(h[5:7],16)
    return f"rgba({r},{g},{b},{a})"

def fg_bar(score):
    lb, col = fear_greed_label(score)
    return f"""
    <div style="margin:6px 0 2px 0;">
        <div style="position:relative;height:10px;border-radius:5px;
            background:linear-gradient(to right,#ef4444 0%,#f97316 25%,
            #eab308 50%,#84cc16 75%,#22c55e 100%);">
            <div style="position:absolute;left:{score}%;transform:translateX(-50%);
                top:-5px;width:4px;height:20px;background:white;
                border-radius:2px;box-shadow:0 0 6px rgba(255,255,255,0.8);">
            </div>
        </div>
        <div style="display:flex;justify-content:space-between;
            font-size:0.62rem;color:{MUTED};margin-top:5px;">
            <span>Extreme Fear</span>
            <span style="color:{col};font-weight:700;">{int(score)} — {lb}</span>
            <span>Extreme Greed</span>
        </div>
    </div>"""

def chart_base(fig, h=260):
    fig.update_layout(
        paper_bgcolor=NAVY2, plot_bgcolor=NAVY2,
        margin=dict(l=8,r=8,t=8,b=8), height=h,
        showlegend=False, font=dict(color=MUTED, size=10)
    )
    fig.update_xaxes(showgrid=False, color=MUTED, tickfont=dict(size=9))
    fig.update_yaxes(showgrid=True,  gridcolor=NAVY3,
                     color=MUTED, tickfont=dict(size=9), zeroline=False)
    return fig

def period_sel(key):
    opts = {"7D":7,"30D":30,"90D":90,"180D":180,"1Y":365}
    if key not in st.session_state:
        st.session_state[key] = "90D"
    cols = st.columns(len(opts))
    for i,(lbl,_) in enumerate(opts.items()):
        with cols[i]:
            if st.button(lbl, key=f"p_{key}_{lbl}",
                         use_container_width=True):
                st.session_state[key] = lbl
                st.rerun()
    return opts[st.session_state[key]]


# ─────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────
def chart_price_rsi(df, coin, days, color):
    sub = df[df["coin"]==coin].tail(days)
    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,
                        row_heights=[0.65,0.35],vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=sub["timestamp"],y=sub["price"],
        mode="lines",line=dict(color=color,width=2),
        fill="tozeroy",fillcolor=hex_rgba(color,0.08)),row=1,col=1)
    fig.add_trace(go.Scatter(x=sub["timestamp"],y=sub["rsi_14"],
        mode="lines",line=dict(color=PURPLE,width=1.5)),row=2,col=1)
    fig.add_hline(y=70,line_dash="dash",line_color=RED,  opacity=0.4,row=2,col=1)
    fig.add_hline(y=30,line_dash="dash",line_color=GREEN,opacity=0.4,row=2,col=1)
    fig.add_hline(y=50,line_dash="dot", line_color=MUTED,opacity=0.3,row=2,col=1)
    fig = chart_base(fig, h=320)
    fig.update_yaxes(tickformat="$,.0f",row=1,col=1)
    fig.update_yaxes(range=[0,100],     row=2,col=1)
    return fig

def chart_macd(df, coin, days):
    sub    = df[df["coin"]==coin].tail(days)
    colors = [GREEN if v>=0 else RED for v in sub["macd_histogram"]]
    fig    = go.Figure()
    fig.add_trace(go.Bar(x=sub["timestamp"],y=sub["macd_histogram"],
                         marker_color=colors))
    fig.add_hline(y=0,line_color=MUTED,line_width=0.8,opacity=0.5)
    return chart_base(fig, h=220)

def chart_bb(df, coin, days, color):
    sub   = df[df["coin"]==coin].tail(days)
    ma14  = sub["price"].rolling(14).mean()
    bstd  = sub["price"].rolling(14).std()
    upper = ma14 + 2*bstd
    lower = ma14 - 2*bstd
    fig   = go.Figure()
    fig.add_trace(go.Scatter(x=sub["timestamp"],y=upper,
        mode="lines",line=dict(color=MUTED,width=1,dash="dot"),name="Upper"))
    fig.add_trace(go.Scatter(x=sub["timestamp"],y=lower,
        mode="lines",line=dict(color=MUTED,width=1,dash="dot"),
        fill="tonexty",fillcolor=hex_rgba("94a3b8",0.05),name="Lower"))
    fig.add_trace(go.Scatter(x=sub["timestamp"],y=sub["price"],
        mode="lines",line=dict(color=color,width=2),name="Price"))
    fig.add_trace(go.Scatter(x=sub["timestamp"],y=ma14,
        mode="lines",line=dict(color=AMBER,width=1.2,dash="dash"),name="MA14"))
    fig = chart_base(fig, h=250)
    fig.update_yaxes(tickformat="$,.0f")
    return fig

def chart_vol(df, coin, days):
    sub = df[df["coin"]==coin].tail(days)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["timestamp"],y=sub["volatility_7d"]*100,
        mode="lines",line=dict(color=BTC,width=1.8),
        fill="tozeroy",fillcolor=hex_rgba(BTC,0.08),name="7d"))
    fig.add_trace(go.Scatter(x=sub["timestamp"],y=sub["volatility_21d"]*100,
        mode="lines",line=dict(color=ETH,width=1.2,dash="dash"),name="21d"))
    fig = chart_base(fig, h=200)
    fig.update_yaxes(ticksuffix="%")
    return fig

def chart_volume(df, coin, days, color):
    sub = df[df["coin"]==coin].tail(days)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=sub["timestamp"],y=sub["volume"],
        marker_color=hex_rgba(color,0.6),
        marker_line_color=color,marker_line_width=0.5))
    fig = chart_base(fig, h=180)
    fig.update_yaxes(tickformat=".2s")
    return fig


# ─────────────────────────────────────────────────────────────
# SMART REFRESH
# ─────────────────────────────────────────────────────────────
def smart_refresh():
    if os.path.exists(FEATURES_PATH):
        age = (time.time() - os.path.getmtime(FEATURES_PATH)) / 3600
        if age <= REFRESH_HOURS:
            return
    with st.spinner("🔄 Updating market data... (once per day)"):
        try:
            subprocess.run(["python", PIPELINE_PATH],
                           capture_output=True, text=True, timeout=120)
        except Exception as e:
            st.warning(f"⚠️ Data update failed: {e} — using cached data")


# ─────────────────────────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_df():
    return pd.read_csv(FEATURES_PATH, parse_dates=["timestamp"])

@st.cache_data(ttl=1800, show_spinner=False)
def get_pred(coin):  return predict(coin)

@st.cache_data(ttl=1800, show_spinner=False)
def get_news(coin):  return fetch_news(coin)


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
for k,v in [("page","home"),
            ("chat_history",[]),("preds",{}),("news",{})]:
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────
# BOOT
# ─────────────────────────────────────────────────────────────
smart_refresh()
df = load_df()

with st.spinner("⚙️ Loading AI predictions..."):
    for coin in ["btc","eth"]:
        if coin not in st.session_state.preds:
            st.session_state.preds[coin] = get_pred(coin)
        if coin not in st.session_state.news:
            st.session_state.news[coin]  = get_news(coin)

btc_pred = st.session_state.preds["btc"]
eth_pred = st.session_state.preds["eth"]
btc_feat = df[df["coin"]=="btc"].iloc[-1]
eth_feat = df[df["coin"]=="eth"].iloc[-1]

last_upd = datetime.fromtimestamp(
    os.path.getmtime(FEATURES_PATH)
).strftime("%b %d, %Y %H:%M")


# ─────────────────────────────────────────────────────────────
# NAVBAR
# ─────────────────────────────────────────────────────────────
def navbar():
    c1, c2, c3 = st.columns([3,4,3])
    with c1:
        st.markdown(f"""
        <div style="padding-top:4px;">
            <span style="font-size:1.6rem;font-weight:800;
                background:linear-gradient(90deg,{BTC},{ETH});
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                CoinTrend
            </span>
            <div style="font-size:0.68rem;color:{MUTED};
                letter-spacing:1.5px;margin-top:2px;">
                ONE STEP AHEAD OF THE MARKET
            </div>
        </div>""", unsafe_allow_html=True)
    
    with c2:
        n1,n2,n3 = st.columns(3)
        pages = [("🏠 Home","home"),("₿ Bitcoin","btc"),("Ξ Ethereum","eth")]
        for col,(lbl,pg) in zip([n1,n2,n3], pages):
            with col:
                t = "primary" if st.session_state.page==pg else "secondary"
                if st.button(lbl, key=f"nav_{pg}", type=t,
                             use_container_width=True):
                    st.session_state.page = pg
                    st.rerun()
    with c3:
        st.markdown(f"""
        <div style="text-align:right;padding-top:6px;">
            <div style="font-size:0.75rem;color:{MUTED};">
                📅 {datetime.now().strftime("%b %d, %Y")}
            </div>
            <div style="font-size:0.68rem;color:{MUTED};margin-top:2px;">
                Updated: {last_upd}
            </div>
        </div>""", unsafe_allow_html=True)
    st.markdown(f"<hr style='margin:8px 0 20px 0;'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────────────────────
def page_home():
    st.markdown('<div class="section-title">🌍 Global Market Indicators</div>',
                unsafe_allow_html=True)

    fg  = float(btc_feat["fear_greed"])
    vix = float(btc_feat["vix"])
    dxy = float(btc_feat["dxy_return_ma7"])
    spy = float(btc_feat["spy_return_ma7"])
    bull  = btc_feat["bull_bear_flag"] == 1
    volhi = btc_feat["volatility_regime"] == 1

    _,fgc = fear_greed_label(fg)
    g1,g2,g3,g4,g5,g6 = st.columns(6)

    with g1:
        st.markdown(f"""
        <div class="card">
            <div class="card-label">Fear & Greed</div>
            <div class="card-value" style="color:{fgc};">{int(fg)}</div>
            {fg_bar(fg)}
        </div>""", unsafe_allow_html=True)

    with g2:
        bc = GREEN if bull else RED
        st.markdown(f"""
        <div class="card">
            <div class="card-label">Market Trend</div>
            <div class="card-value" style="color:{bc};">
                {"Bull 🐂" if bull else "Bear 🐻"}
            </div>
            <div class="card-sub">BTC vs MA50</div>
        </div>""", unsafe_allow_html=True)

    with g3:
        vc = RED if vix>25 else GREEN
        st.markdown(f"""
        <div class="card">
            <div class="card-label">VIX</div>
            <div class="card-value" style="color:{vc};">{vix:.1f}</div>
            <div class="card-sub">{"High Fear ⚠️" if vix>25 else "Low Fear ✅"}</div>
        </div>""", unsafe_allow_html=True)

    with g4:
        dc = RED if dxy>0 else GREEN
        st.markdown(f"""
        <div class="card">
            <div class="card-label">DXY Trend (7d)</div>
            <div class="card-value" style="color:{dc};">
                {"↑ Strong" if dxy>0 else "↓ Weak"}
            </div>
            <div class="card-sub">
                {"Bearish for crypto" if dxy>0 else "Bullish for crypto"}
            </div>
        </div>""", unsafe_allow_html=True)

    with g5:
        sc = GREEN if spy>0 else RED
        st.markdown(f"""
        <div class="card">
            <div class="card-label">SPY Trend (7d)</div>
            <div class="card-value" style="color:{sc};">
                {"↑ Risk-On" if spy>0 else "↓ Risk-Off"}
            </div>
            <div class="card-sub">
                {"Bullish for crypto" if spy>0 else "Bearish for crypto"}
            </div>
        </div>""", unsafe_allow_html=True)

    with g6:
        vlc = RED if volhi else GREEN
        st.markdown(f"""
        <div class="card">
            <div class="card-label">Volatility Regime</div>
            <div class="card-value" style="color:{vlc};">
                {"High ⚡" if volhi else "Low 😴"}
            </div>
            <div class="card-sub">
                {"Expect large moves" if volhi else "Calm market"}
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Assets table
    st.markdown('<div class="section-title">💎 Assets</div>',
                unsafe_allow_html=True)

    hcols = st.columns([2,2,2,2,2,2,1])
    for hc, lbl in zip(hcols,
        ["Asset","Price","Tomorrow's Direction",
         "Confidence","Volatility 7d","RSI",""]):
        with hc:
            st.markdown(f"""
            <div style="font-size:0.70rem;color:{MUTED};text-transform:uppercase;
                        letter-spacing:0.08em;padding:4px 0 8px 0;
                        border-bottom:1px solid {BORDER};">
                {lbl}
            </div>""", unsafe_allow_html=True)

    for coin, pred, feat, color, name, sym in [
        ("btc", btc_pred, btc_feat, BTC, "Bitcoin",  "₿"),
        ("eth", eth_pred, eth_feat, ETH, "Ethereum", "Ξ"),
    ]:
        is_up  = pred["direction"].startswith("UP")
        dc     = GREEN if is_up else RED
        cc     = conf_color(pred["confidence"])
        rsi    = float(feat["rsi_14"])
        ri,rc  = rsi_label(rsi)
        vol    = float(feat["volatility_7d"])*100
        price  = float(feat["price"])
        vc     = RED if vol>3 else GREEN

        r1,r2,r3,r4,r5,r6,r7 = st.columns([2,2,2,2,2,2,1])
        with r1:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;padding:12px 0;">
                <div style="width:36px;height:36px;border-radius:50%;
                    background:{color}22;display:flex;align-items:center;
                    justify-content:center;font-size:1.1rem;color:{color};">
                    {sym}
                </div>
                <div>
                    <div style="font-weight:700;">{name}</div>
                    <div style="color:{MUTED};font-size:0.75rem;">{coin.upper()}</div>
                </div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div style="padding:12px 0;font-weight:700;font-size:1rem;">
                ${price:,.2f}
            </div>""", unsafe_allow_html=True)
        with r3:
            st.markdown(f"""
            <div style="padding:12px 0;color:{dc};font-weight:700;font-size:1rem;">
                {pred['direction']}
            </div>""", unsafe_allow_html=True)
        with r4:
            st.markdown(f"""
            <div style="padding:12px 0;">
                <div style="color:{cc};font-weight:700;">{pred['confidence']}%</div>
                <div style="background:{NAVY3};border-radius:3px;
                    height:4px;margin-top:4px;">
                    <div style="background:{cc};width:{pred['confidence']}%;
                        height:4px;border-radius:3px;"></div>
                </div>
            </div>""", unsafe_allow_html=True)
        with r5:
            st.markdown(f"""
            <div style="padding:12px 0;color:{vc};font-weight:600;">
                {vol:.2f}%
            </div>""", unsafe_allow_html=True)
        with r6:
            st.markdown(f"""
            <div style="padding:12px 0;">
                <span style="color:{rc};font-weight:600;">{rsi:.1f}</span>
                <span style="color:{MUTED};font-size:0.75rem;
                    margin-left:6px;">{ri}</span>
            </div>""", unsafe_allow_html=True)
        with r7:
            if st.button("View →", key=f"view_{coin}",
                         use_container_width=True):
                st.session_state.page = coin
                st.rerun()
        st.markdown(f"<hr style='margin:0;'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# COIN DETAIL PAGE
# ─────────────────────────────────────────────────────────────
def page_coin(coin):
    pred  = st.session_state.preds[coin]
    feat  = df[df["coin"]==coin].iloc[-1]
    color = BTC if coin=="btc" else ETH
    name  = "Bitcoin" if coin=="btc" else "Ethereum"

    is_up = pred["direction"].startswith("UP")
    dc    = GREEN if is_up else RED
    cc    = conf_color(pred["confidence"])

    left, right = st.columns([2,3])

    with left:
        st.markdown(f"""
        <div class="pred-card">
            <div style="color:{MUTED};font-size:0.75rem;
                text-transform:uppercase;letter-spacing:0.1em;">
                {name} — Tomorrow's Direction
            </div>
            <div style="font-size:3.2rem;font-weight:800;
                color:{dc};margin:10px 0;">
                {pred['direction']}
            </div>
            <div style="font-size:1.1rem;color:{cc};font-weight:700;
                margin-bottom:14px;">
                {pred['confidence']}% confidence
            </div>
            <div style="background:{NAVY3};border-radius:6px;
                height:8px;margin:8px 0;">
                <div style="background:{cc};width:{pred['confidence']}%;
                    height:8px;border-radius:6px;"></div>
            </div>
            <div style="font-size:0.73rem;color:{MUTED};margin-top:14px;">
                Based on technical, macro &amp; sentiment signals
            </div>
            <div style="font-size:0.70rem;color:{MUTED};
                margin-top:4px;font-style:italic;">
                ⚠️ Not financial advice
            </div>
        </div>""", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-title">📊 Indicators</div>',
                    unsafe_allow_html=True)

        rsi  = float(feat["rsi_14"])
        macd = float(feat["macd_histogram"])
        bb   = float(feat["bb_pct"])
        mom  = float(feat["momentum_acceleration"])

        ri,rc  = rsi_label(rsi)
        mi,mc  = macd_label(macd)
        bi,bbc = bb_label(bb)
        mo,moc = mom_label(mom)

        ia,ib = st.columns(2)
        ic,id_ = st.columns(2)

        for col, lbl, val, interp, icol in [
            (ia, "RSI 14",       f"{rsi:.1f}",  ri, rc),
            (ib, "MACD Signal",  f"{macd:.2f}", mi, mc),
        ]:
            with col:
                st.markdown(f"""
                <div class="card">
                    <div class="card-label">{lbl}</div>
                    <div class="card-value" style="color:{icol};">{val}</div>
                    <div class="card-sub" style="color:{icol};">{interp}</div>
                </div>""", unsafe_allow_html=True)

        for col, lbl, val, interp, icol in [
            (ic,  "BB Position", f"{bb:.2f}",   bi, bbc),
            (id_, "Momentum",    f"{mom:.4f}",  mo, moc),
        ]:
            with col:
                st.markdown(f"""
                <div class="card">
                    <div class="card-label">{lbl}</div>
                    <div class="card-value" style="color:{icol};">{val}</div>
                    <div class="card-sub" style="color:{icol};">{interp}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Price + RSI
    st.markdown('<div class="section-title">📈 Price & RSI 14</div>',
                unsafe_allow_html=True)
    d1 = period_sel(f"pr_{coin}")
    st.plotly_chart(chart_price_rsi(df, coin, d1, color),
                    use_container_width=True,
                    config={"displayModeBar":False},
                    key=f"cpr_{coin}")

    cl, cr = st.columns(2)
    with cl:
        st.markdown('<div class="section-title">📊 MACD Histogram</div>',
                    unsafe_allow_html=True)
        d2 = period_sel(f"macd_{coin}")
        st.plotly_chart(chart_macd(df, coin, d2),
                        use_container_width=True,
                        config={"displayModeBar":False},
                        key=f"cmacd_{coin}")
    with cr:
        st.markdown('<div class="section-title">📉 Bollinger Bands</div>',
                    unsafe_allow_html=True)
        d3 = period_sel(f"bb_{coin}")
        st.plotly_chart(chart_bb(df, coin, d3, color),
                        use_container_width=True,
                        config={"displayModeBar":False},
                        key=f"cbb_{coin}")

    cl2, cr2 = st.columns(2)
    with cl2:
        st.markdown('<div class="section-title">⚡ Volatility</div>',
                    unsafe_allow_html=True)
        d4 = period_sel(f"vol_{coin}")
        st.plotly_chart(chart_vol(df, coin, d4),
                        use_container_width=True,
                        config={"displayModeBar":False},
                        key=f"cvol_{coin}")
    with cr2:
        st.markdown('<div class="section-title">📦 Volume</div>',
                    unsafe_allow_html=True)
        d5 = period_sel(f"volume_{coin}")
        st.plotly_chart(chart_volume(df, coin, d5, color),
                        use_container_width=True,
                        config={"displayModeBar":False},
                        key=f"cvolume_{coin}")

# ─────────────────────────────────────────────────────────────
# FLOATING CHAT 
# ─────────────────────────────────────────────────────────────

QUICK = [
    ("₿ BTC tomorrow?",      "What will Bitcoin do tomorrow?",          "btc"),
    ("Ξ ETH tomorrow?",      "What will Ethereum do tomorrow?",         "eth"),
    ("⚠️ BTC risks today?",  "What are the main risks for BTC today?",  "btc"),
    ("🌍 Market overview?",  "Give me a full crypto market overview",   "btc"),
]

def floating_chat():

    # ── Initialize session state safely
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Centered layout
    _, center, _ = st.columns([1, 4, 1])

    with center:

        # ── Header
        st.markdown(f"""
        <div style="text-align:center;margin-bottom:20px;">
            <div style="display:inline-flex;flex-direction:column;
                align-items:center;gap:8px;background:{NAVY2};
                border:1px solid {BTC}44;border-radius:20px;
                padding:16px 32px;">
                <div style="width:40px;height:40px;border-radius:12px;
                    background:linear-gradient(135deg,{BTC},{ETH});
                    display:flex;align-items:center;justify-content:center;
                    font-size:1.2rem;">✦</div>
                <div style="font-size:1.0rem;font-weight:700;color:{TEXT};">
                    CoinTrend AI Analyst
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ─────────────────────────────────────────
        # CHAT HISTORY
        # ─────────────────────────────────────────
        if st.session_state.chat_history:
            for msg in st.session_state.chat_history[-8:]:

                if msg["role"] == "user":
                    st.markdown(f"""
                    <div style="background:#1d4ed8;
                        border-radius:18px 18px 4px 18px;
                        padding:10px 16px;margin:8px 0;
                        max-width:75%;margin-left:auto;
                        font-size:0.85rem;">
                        {msg['content']}
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    st.markdown(f"""
                    <div style="background:{NAVY2};
                        border:1px solid {BORDER};
                        border-radius:18px 18px 18px 4px;
                        padding:12px 16px;margin:8px 0;
                        font-size:0.85rem;line-height:1.6;">
                        {msg['content']}
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

        # ─────────────────────────────────────────
        # INPUT (ENTER SEND ENABLED)
        # ─────────────────────────────────────────
        st.markdown("""
        <style>
        input[type="text"] {
            caret-color: light blue !important;  
        }
        </style>
        """, unsafe_allow_html=True)
        user_input = st.text_input(
            "Ask...",
            placeholder="Ask me anything about BTC, ETH or market conditions...",
            key="chat_input",
            label_visibility="collapsed"
        )

        # ENTER triggers automatically because text_input updates state
        send = bool(user_input)

        # ─────────────────────────────────────────
        # QUICK QUESTIONS
        # ─────────────────────────────────────────

        st.markdown(f"""
        <div style="margin-top:10px;font-size:0.68rem;color:{MUTED};
            text-transform:uppercase;letter-spacing:0.08em;
            margin-bottom:6px;text-align:center;">
            Quick Questions
        </div>
        """, unsafe_allow_html=True)

        qcols = st.columns(4)

        for i, (lbl, q, coin) in enumerate(QUICK):
            with qcols[i]:
                if st.button(lbl, key=f"q_{i}", use_container_width=True):

                    st.session_state.chat_history.append(
                        {"role": "user", "content": q, "coin": coin}
                    )

                    with st.spinner("Analyzing..."):
                        nd   = st.session_state.news.get(coin, get_news(coin))
                        pd_  = st.session_state.preds.get(coin, get_pred(coin))
                        arts = retrieve(q, nd["all_articles"], coin)
                        ctx  = format_context(arts, coin, pd_)
                        resp = analyze(q, ctx)

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": resp}
                    )

                    st.rerun()

        # ─────────────────────────────────────────
        # CLEAR CHAT
        # ─────────────────────────────────────────

        if st.session_state.chat_history:
            _, clr, _ = st.columns([4, 2, 4])
            with clr:
                if st.button("🗑️ Clear", key="clear_chat", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()

    # ─────────────────────────────────────────
    # HANDLE SEND (SAFE STATE CLEAR)
    # ─────────────────────────────────────────

    if send:

        coin = "eth" if any(
            w in user_input.lower()
            for w in ["eth", "ethereum", "ether"]
        ) else "btc"

        st.session_state.chat_history.append(
            {"role": "user", "content": user_input, "coin": coin}
        )

        with st.spinner("Analyzing..."):
            nd   = st.session_state.news.get(coin, get_news(coin))
            pd_  = st.session_state.preds.get(coin, get_pred(coin))
            arts = retrieve(user_input, nd["all_articles"], coin)
            ctx  = format_context(arts, coin, pd_)
            resp = analyze(user_input, ctx)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": resp}
        )

        # SAFE CLEAR (NO STREAMLIT ERROR)
        del st.session_state["chat_input"]

        st.rerun()


# ─────────────────────────────────────────────────────────────
# RENDER
# ─────────────────────────────────────────────────────────────
navbar()

if   st.session_state.page == "home": page_home()
elif st.session_state.page == "btc":  page_coin("btc")
elif st.session_state.page == "eth":  page_coin("eth")

floating_chat()

st.markdown(f"""
<div style="text-align:center;color:{MUTED};font-size:0.72rem;
    margin-top:40px;padding:20px;border-top:1px solid {BORDER};">
    ⚠️ CoinTrend is an AI-powered decision support tool.
    This is <strong>not financial advice</strong>.
    Always do your own research.<br><br>
    CoinTrend © 2026 — One step ahead of the market
</div>""", unsafe_allow_html=True)