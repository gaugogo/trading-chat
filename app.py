#!/usr/bin/env python3
"""
Trading Chat · Multi-Timeframe Analysis
"""

import sys, os, time
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from instruments import INSTRUMENTS
from core import fetch_all_timeframes, adjust_to_spot, TF_ORDER, fmt_price
from position import build_position_data_for_ai
from swing import build_swing_data_for_ai
from daytrade import build_daytrade_data_for_ai
from scalp import build_scalp_data_for_ai
from analysis import call_deepseek, build_raw_data_summary

# ─── Config ────────────────────────────────────────────────────────

INSTRUMENT_KEYS = list(INSTRUMENTS.keys())
TF_NAMES = ["Daily", "4H", "1H", "15m", "5m"]

STRATEGY_LABELS = ["Position", "Swing", "Daytrade", "Scalp", "Ichimoku", "SMC", "PA", "Data"]
STRATEGY_KEYS   = ["position", "swing", "daytrade", "scalp", "ichimoku", "smc", "pa", "data"]
STRATEGY_MAP    = dict(zip(STRATEGY_KEYS, STRATEGY_LABELS))

AI_MODELS = ["DeepSeek Chat", "DeepSeek Reasoner"]

st.set_page_config(
    page_title="Trading Chat",
    page_icon="chart_with_upwards_trend",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── CSS ────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', -apple-system, sans-serif; }

    .stApp { max-width: 820px; margin: 0 auto; }

    .block-container { padding: 2rem 1.2rem !important; }

    /* ── Radio (strategy pills) ── */
    div[data-testid="stRadio"] > label { display: none; }
    div[data-testid="stRadio"] > div {
        display: flex; flex-wrap: wrap; gap: 4px;
    }
    div[data-testid="stRadio"] > div label {
        padding: 6px 14px; border-radius: 20px; font-size: 0.82rem; font-weight: 500;
        border: 1.5px solid #2a2a3a; background: transparent; cursor: pointer;
        transition: all 0.15s; letter-spacing: -0.2px; color: #8b8ba0;
    }
    div[data-testid="stRadio"] > div label:hover {
        border-color: #4a4a6a; color: #c0c0d0;
    }
    div[data-testid="stRadio"] > div label[data-selected="true"] {
        background: #2d2d50; border-color: #6b6bff; color: #a0b8ff; font-weight: 600;
    }

    /* ── Buttons ── */
    .stButton > button {
        border-radius: 10px; font-weight: 500; font-size: 0.84rem;
        padding: 0.45rem 1rem; transition: all 0.15s; letter-spacing: -0.2px;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #5b6dff, #4a57cc);
        border: none; color: #fff;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #6b7dff, #5a67dd);
    }
    .stButton > button[kind="secondary"] {
        background: #1e1e2e; border: 1.5px solid #2a2a3a; color: #a0a0b8;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: #4a4a6a; color: #c0c0d0;
    }

    /* ── Select widgets ── */
    div[data-testid="stSelectbox"] label { display: none; }
    .stSelectbox > div > div {
        border-radius: 8px !important; border: 1.5px solid #2a2a3a !important;
        background: #0e0e18 !important;
    }

    /* ── Price card ── */
    .price-card {
        background: linear-gradient(145deg, #111122 0%, #161630 100%);
        border: 1px solid #252540; border-radius: 16px;
        padding: 1.2rem 1.5rem; margin-bottom: 1.2rem;
        position: relative; overflow: hidden;
    }
    .price-card::after {
        content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, #6b6bff44, #6b6bff, #6b6bff44);
    }
    .price-value {
        font-size: 2.4rem; font-weight: 700; letter-spacing: -1px;
        font-feature-settings: 'tnum'; color: #f0f0f8;
    }
    .price-change { font-size: 0.9rem; font-weight: 500; }
    .price-green { color: #4ecb71; }
    .price-red { color: #e0556a; }
    .price-meta { font-size: 0.78rem; color: #6a6a88; margin-top: 0.2rem; }

    /* ── Section title ── */
    .section-title {
        font-size: 0.8rem; font-weight: 600; letter-spacing: 0.5px;
        text-transform: uppercase; color: #5a5a7a; margin: 1rem 0 0.4rem;
    }

    /* ── Result ── */
    .result-block {
        background: #111122; border: 1px solid #22223a; border-radius: 12px;
        padding: 1.2rem 1.5rem; font-size: 0.85rem; line-height: 1.5;
        color: #c0c0d8; font-family: 'JetBrains Mono', monospace;
        white-space: pre-wrap; overflow-x: auto;
    }

    /* ── Stats bar ── */
    .stat-row {
        display: flex; gap: 8px; margin-top: 12px;
    }
    .stat-item {
        flex: 1; background: #111122; border: 1px solid #22223a;
        border-radius: 10px; padding: 10px 12px; text-align: center;
    }
    .stat-value { font-size: 1.05rem; font-weight: 600; color: #d0d0e8; font-feature-settings: 'tnum'; }
    .stat-label { font-size: 0.7rem; color: #5a5a7a; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.3px; }

    /* ── Divider ── */
    .hr { margin: 1.2rem 0; border: none; height: 1px; background: #1e1e36; }

    /* ── Status tags ── */
    .tag { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.7rem; font-weight: 600; }
    .tag-ok { background: #16281e; color: #4ecb71; }
    .tag-warn { background: #281616; color: #e0556a; }

    /* ── Responsive ── */
    @media (max-width: 640px) {
        .price-value { font-size: 1.8rem; }
        .block-container { padding: 1rem 0.8rem !important; }
    }
</style>
""", unsafe_allow_html=True)

# ─── Session ────────────────────────────────────────────────────────

_DEFAULTS = {
    "instrument": "xau",
    "tf_data": None,
    "last_result": "",
    "ai_response": "",
    "active_strategy": "position",
    "ai_model": AI_MODELS[0],
    "last_fetch_time": None,
}

for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Helpers ────────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def _cached_fetch(instrument: str) -> Optional[Dict[str, pd.DataFrame]]:
    if instrument not in INSTRUMENTS:
        return None
    try:
        tf = fetch_all_timeframes(INSTRUMENTS[instrument], use_cache=True)
        return adjust_to_spot(tf, INSTRUMENTS[instrument]) if tf else None
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu: {e}")
        return None

def _spot(tf_data: Dict[str, pd.DataFrame], cfg: Dict[str, Any]) -> Tuple[float, str]:
    for t in ["1H", "4H", "Daily", "15m", "5m"]:
        df = tf_data.get(t)
        if df is not None and not df.empty:
            return float(df.iloc[-1]["Close"]), t
    return 0.0, ""

def _fmt(v: Any, d: int) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{v:.{d}f}"

def _build_ai_data(key: str, instr: str) -> str:
    """Build raw data for AI based on strategy."""
    tf = st.session_state.tf_data
    cfg = INSTRUMENTS[instr]
    if not tf:
        return "Lỗi: không có dữ liệu"
    builders = {
        "position": build_position_data_for_ai,
        "swing": build_swing_data_for_ai,
        "daytrade": build_daytrade_data_for_ai,
        "scalp": build_scalp_data_for_ai,
    }
    fn = builders.get(key)
    if fn:
        return fn(tf, cfg)
    # fallback: generic raw data
    return build_raw_data_summary(tf, cfg)

def _chart(tf_data: Dict[str, pd.DataFrame], tf_name: str) -> go.Figure:
    df = tf_data.get(tf_name)
    if df is None or df.empty or len(df) < 5:
        fig = go.Figure()
        fig.add_annotation(text="Không đủ dữ liệu", showarrow=False, font=dict(size=18, color="#3a3a58"))
        fig.update_layout(template="plotly_dark", height=320, paper_bgcolor="#0d0d18", plot_bgcolor="#0d0d18")
        return fig

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.03)

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="", showlegend=False,
        increasing=dict(line=dict(color="#4ecb71", width=1), fillcolor="rgba(78,203,113,0.15)"),
        decreasing=dict(line=dict(color="#e0556a", width=1), fillcolor="rgba(224,85,106,0.15)"),
    ), row=1, col=1)

    for col, color, name in [("SMA_20", "#7b8cde", "SMA20"), ("SMA_50", "#c8a45c", "SMA50")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], line=dict(color=color, width=1), name=name), row=1, col=1)

    if "SMA_200" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_200"], line=dict(color="#8e6fc4", width=1, dash="dot"), name="SMA200"), row=1, col=1)

    if "Volume" in df.columns:
        vcolors = ["rgba(224,85,106,0.4)" if df["Close"].iloc[i] < df["Open"].iloc[i] else "rgba(78,203,113,0.4)" for i in range(len(df))]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=vcolors, name="", showlegend=False), row=2, col=1)

    if "RSI_14" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI_14"], line=dict(color="#6bc5d9", width=1.5), name="RSI", showlegend=False), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(224,85,106,0.267)", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(78,203,113,0.267)", row=2, col=1)
        fig.update_yaxes(range=[0, 100], row=2, col=1, showgrid=False)

    fig.update_layout(
        template="plotly_dark", height=440, margin=dict(l=0, r=0, t=0, b=0),
        hovermode="x unified", font=dict(size=9, color="#8b8ba0"),
        paper_bgcolor="#0d0d18", plot_bgcolor="#0d0d18",
        legend=dict(orientation="h", yanchor="top", y=1.12, xanchor="left", x=0, font=dict(size=8)),
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig

# ═══════════════════════════════════════════════════════════════════════

# ─── Header ───
st.markdown("""
<div style="display:flex;align-items:baseline;gap:10px;margin-bottom:0.8rem">
    <span style="font-size:1.5rem;font-weight:700;color:#c0c0e0;letter-spacing:-0.5px">trading</span>
    <span style="font-size:1.5rem;font-weight:300;color:#5a5a7a">chat</span>
</div>
""", unsafe_allow_html=True)

# ─── Top bar: instrument + time ───
col1, col2, col3 = st.columns([2, 1, 1.5])
with col1:
    instr = st.selectbox(
        "Sản phẩm",
        INSTRUMENT_KEYS,
        format_func=lambda x: f"{INSTRUMENTS[x]['display_name']}  ·  {x.upper()}",
        label_visibility="collapsed",
        key="instrument",
    )
with col2:
    refresh = st.button("🔄 Làm mới", type="secondary", use_container_width=True, key="btn_refresh")
    if refresh:
        st.cache_data.clear()
        st.session_state.tf_data = None
        st.rerun()
with col3:
    clear = st.button("🗑 Xoá cache", type="secondary", use_container_width=True, key="btn_clear")
    if clear:
        st.cache_data.clear()
        st.session_state.tf_data = None
        st.session_state.last_result = ""
        # Also clear file cache
        import glob, os as _os
        for f in glob.glob("cache/*.json"):
            try:
                _os.remove(f)
            except Exception:
                pass
        st.rerun()

# ─── Data ───
with st.spinner("Đang tải dữ liệu..."):
    try:
        fresh = _cached_fetch(instr)
        if fresh:
            st.session_state.tf_data = fresh
            st.session_state.last_fetch_time = datetime.now(timezone.utc)
        elif not st.session_state.tf_data:
            st.error("Không thể tải dữ liệu. Kiểm tra kết nối mạng.")
            st.stop()
    except Exception as e:
        if not st.session_state.tf_data:
            st.error(f"Lỗi kết nối: {e}")
            st.stop()

tf_data = st.session_state.tf_data
if not tf_data:
    st.error("Không có dữ liệu.")
    st.stop()

cfg = INSTRUMENTS[instr]
d = cfg["decimals"]

# ─── Price ───
price, tf_name = _spot(tf_data, cfg)
if price > 0:
    src_df = tf_data.get(tf_name)
    prev = float(src_df.iloc[-2]["Close"]) if src_df is not None and len(src_df) >= 2 else price
    chg = price - prev
    pct = (chg / prev) * 100 if prev > 0 else 0
    css = "price-green" if chg >= 0 else "price-red"
    sig = "+" if chg >= 0 else ""

    st.markdown(f"""
    <div class="price-card">
        <div class="price-value">{_fmt(price, d)}</div>
        <div style="margin-top:4px">
            <span class="price-change {css}">{sig}{_fmt(chg, d)}&nbsp; ({sig}{pct:.2f}%)</span>
        </div>
        <div class="price-meta">
            {cfg['display_name']} &nbsp;·&nbsp; {tf_name} &nbsp;·&nbsp;
            {datetime.now(timezone.utc).strftime('%d/%m/%Y %H:%M UTC')}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Strategy ───
st.markdown('<div class="section-title">Chiến lược</div>', unsafe_allow_html=True)

idx = STRATEGY_KEYS.index(st.session_state.active_strategy)
label = st.radio("Chiến lược", STRATEGY_LABELS, index=idx, label_visibility="collapsed", horizontal=True, key="rd_strat")
key = STRATEGY_KEYS[STRATEGY_LABELS.index(label)]
if key != st.session_state.active_strategy:
    st.session_state.active_strategy = key
    st.session_state.last_result = ""
    st.rerun()

# ─── AI model ───
col_m1, col_m2 = st.columns([1, 2])
with col_m1:
    st.markdown('<div class="section-title">AI model</div>', unsafe_allow_html=True)
with col_m2:
    st.session_state.ai_model = st.selectbox(
        "Model AI", AI_MODELS,
        index=AI_MODELS.index(st.session_state.ai_model) if st.session_state.ai_model in AI_MODELS else 0,
        label_visibility="collapsed", key="sel_model",
    )

# ─── Analyze → DeepSeek ───
col_btn1, col_btn2 = st.columns([1, 1])
with col_btn1:
    analyze = st.button("⚡ Phân tích AI", type="primary", use_container_width=True)
with col_btn2:
    show_raw = st.button("📊 Xem dữ liệu thô", type="secondary", use_container_width=True)

if analyze:
    api_key = st.secrets.get("DEEPSEEK_API_KEY", "") or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        st.error("Thiếu DEEPSEEK_API_KEY. Thêm vào .env hoặc secrets.toml")
    else:
        s_key = st.session_state.active_strategy
        s_label = STRATEGY_MAP.get(s_key, s_key)
        with st.spinner(f"Đang gọi DeepSeek phân tích {s_label}..."):
            raw_data = _build_ai_data(s_key, instr)
            model = st.session_state.ai_model
            # Use thinking model if reasoner selected
            use_thinking = (model == "DeepSeek Reasoner")
            cfg_ai = dict(cfg)
            if use_thinking:
                cfg_ai["deepseek_model"] = "deepseek-reasoner"
                cfg_ai["deepseek_thinking"] = True
            q = (
                "Phân tích dữ liệu trên và cho khuyến nghị giao dịch:\n"
                "1. Bias thị trường + lý do\n"
                "2. Entry/SL/TP + R:R\n"
                "3. Rủi ro & cảnh báo"
            )
            try:
                resp = call_deepseek(raw_data, q, api_key, cfg_ai)
                st.session_state.ai_response = resp if resp else "❌ AI không phản hồi"
            except Exception as e:
                st.session_state.ai_response = f"❌ Lỗi: {e}"
        st.rerun()

if show_raw:
    s_key = st.session_state.active_strategy
    raw_data = _build_ai_data(s_key, instr)
    st.session_state.last_result = raw_data

# ═══════════════════════════════════════════════════════════════════════
#  AI Analysis Result
# ═══════════════════════════════════════════════════════════════════════

if st.session_state.ai_response:
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    r = st.session_state.ai_response
    if r.startswith("❌"):
        st.error(r)
    else:
        st.markdown("##### 🤖 AI Nhận Định")
        st.markdown(f'<div class="result-block">{r}</div>', unsafe_allow_html=True)

# Raw data display (when requested)
if st.session_state.last_result:
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    result = st.session_state.last_result
    is_err = result.startswith("Lỗi")
    if is_err:
        st.error(result)
    else:
        with st.expander("📊 Dữ liệu thô", expanded=False):
            st.markdown(f'<div class="result-block">{result}</div>', unsafe_allow_html=True)

    # ─── Chart ───
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Biểu đồ</div>', unsafe_allow_html=True)

    tf_idx = {"Daily": 0, "4H": 1, "1H": 2, "15m": 3, "5m": 4}
    sel_tf = st.selectbox(
        "Khung thời gian", TF_NAMES,
        index=tf_idx.get(st.session_state.get("last_tf", "1H"), 2),
        label_visibility="collapsed", key="sel_tf",
    )
    st.session_state.last_tf = sel_tf

    df = tf_data.get(sel_tf)
    if df is not None and not df.empty:
        n = len(df)
        rng = f"{df.index[0].strftime('%d/%m/%y')} - {df.index[-1].strftime('%d/%m/%y %H:%M')}"
        st.markdown(f'<p style="font-size:0.75rem;color:#4a4a6a;text-align:right;margin:0 0 -8px">{n} nến · {rng}</p>', unsafe_allow_html=True)

    st.plotly_chart(_chart(tf_data, sel_tf), use_container_width=True, config={
        "displayModeBar": True, "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    })

    # ─── Quick stats ───
    df = tf_data.get(sel_tf)
    if df is not None and len(df) >= 5:
        last = df.iloc[-1]
        def _s(col, fmt_spec=".2f"):
            v = last.get(col, np.nan)
            return f"{float(v):{fmt_spec}}" if not pd.isna(v) else "--"

        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-item"><div class="stat-value">{_fmt(float(last['Close']), d)}</div><div class="stat-label">Đóng</div></div>
            <div class="stat-item"><div class="stat-value">{_fmt(float(last['High']), d)}</div><div class="stat-label">Cao</div></div>
            <div class="stat-item"><div class="stat-value">{_fmt(float(last['Low']), d)}</div><div class="stat-label">Thấp</div></div>
            <div class="stat-item"><div class="stat-value">{_s('RSI_14', '.1f')}</div><div class="stat-label">RSI</div></div>
            <div class="stat-item"><div class="stat-value">{_fmt(float(last.get('ATR',0)), d)}</div><div class="stat-label">ATR</div></div>
        </div>
        """, unsafe_allow_html=True)

    # Show offset info
    offset_info = ""
    if cfg.get("has_spot"):
        from core import fetch_spot_price
        spot = fetch_spot_price(cfg["spot_url"], instrument_id=cfg.get("id", "xau"), symbol=cfg.get("symbol", ""))
        if spot:
            diff = price - spot
            offset_info = f"  ·  Spot: {spot:.{d}f} (futures {diff:+.{d}f})"
        else:
            offset_info = "  ·  Spot: N/A"
    st.caption(
        f"Cập nhật: {st.session_state.last_fetch_time.strftime('%Y-%m-%d %H:%M UTC') if st.session_state.last_fetch_time else 'N/A'}"
        f"  ·  Yahoo Finance  ·  cache 60s{offset_info}"
    )
