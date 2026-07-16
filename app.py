#!/usr/bin/env python3
"""
Trading Chat Web App — Streamlit UI
Chạy trên điện thoại / trình duyệt mobile

Usage:
  streamlit run app.py                         # local
  streamlit run app.py --server.port 8501 --server.address 0.0.0.0   # network
"""

import sys
import os
import json
from io import StringIO
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Project imports ──
from instruments import INSTRUMENTS
from mcp_server import fetch_data
from core import (
    fetch_all_timeframes,
    fetch_spot_price,
    determine_trend,
    build_confluence_summary,
    adjust_to_spot,
    fmt_price,
    TF_ORDER,
)
from regime import detect_regime, regime_recommendation
from divergence import analyze_all_divergences
from position import run_position_analysis, run_position_signal
from swing import run_swing_analysis, run_swing_signal
from daytrade import run_daytrade_analysis, run_daytrade_signal
from scalp import run_scalp_analysis, run_scalp_signal
from ichimoku import run_ichimoku_analysis, run_ichimoku_signal
from analysis import call_deepseek, format_report, build_technical_summary
from smc import analyze_all_smc, format_smc_footer

# ── Page config ──
st.set_page_config(
    page_title="Trading Chat",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Constants ──
INSTRUMENT_KEYS = list(INSTRUMENTS.keys())
TF_NAMES = ["Daily", "4H", "1H", "15m", "5m"]

# ── CSS cho mobile ──
st.markdown("""
<style>
    .stApp { max-width: 100%; }
    .stButton > button { width: 100%; border-radius: 12px; padding: 0.6rem; font-weight: 600; }
    .stSelectbox > div > div { border-radius: 10px; }
    div[data-testid="stExpander"] { border-radius: 12px; }
    .analysis-card {
        background: #1e1e2e; border-radius: 14px; padding: 1.2rem;
        margin-bottom: 1rem; border: 1px solid #313244;
    }
    .metric-value { font-size: 1.6rem; font-weight: 700; }
    .metric-label { font-size: 0.8rem; color: #a6adc8; }
    .signal-buy { color: #a6e3a1; font-weight: 700; }
    .signal-sell { color: #f38ba8; font-weight: 700; }
    .signal-neutral { color: #fab387; font-weight: 700; }
    h1, h2, h3 { margin-bottom: 0.5rem; }
    .stSpinner > div { border-color: #89b4fa !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──
if "instrument" not in st.session_state:
    st.session_state.instrument = "xau"
if "tf_data" not in st.session_state:
    st.session_state.tf_data = None
if "last_result" not in st.session_state:
    st.session_state.last_result = ""
if "active_strategy" not in st.session_state:
    st.session_state.active_strategy = "position"

# ── Caching ──

@st.cache_data(ttl=30)
def cached_fetch_data(instrument: str) -> str:
    return fetch_data(instrument)

@st.cache_data(ttl=60)
def cached_fetch_timeframes(instrument: str):
    if instrument not in INSTRUMENTS:
        return None
    cfg = INSTRUMENTS[instrument]
    tf_data = fetch_all_timeframes(cfg, use_cache=True)
    if tf_data:
        tf_data = adjust_to_spot(tf_data, cfg)
    return tf_data

# ── Helpers ──

def get_latest_price(tf_data) -> float:
    if not tf_data:
        return 0.0
    for tf in TF_NAMES:
        if tf in tf_data and not tf_data[tf].empty:
            return float(tf_data[tf].iloc[-1]["Close"])
    return 0.0

def fmt_price_st(val, d=2):
    if val is None:
        return "N/A"
    try:
        if np.isnan(val):
            return "N/A"
    except:
        pass
    return f"{val:.{d}f}"

def plot_candlestick(df: pd.DataFrame, tf_name: str):
    if df.empty or len(df) < 5:
        return None

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f"{tf_name}", "RSI (14)", "MACD"),
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name="OHLC", increasing_line_color="#a6e3a1",
            decreasing_line_color="#f38ba8",
        ),
        row=1, col=1,
    )

    if "SMA_20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"],
            line=dict(color="#89b4fa", width=1.5), name="SMA 20"), row=1, col=1)
    if "SMA_50" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"],
            line=dict(color="#f9e2af", width=1.5), name="SMA 50"), row=1, col=1)
    if "SMA_200" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_200"],
            line=dict(color="#cba6f7", width=1.5), name="SMA 200"), row=1, col=1)
    if "BB_Upper" in df.columns and "BB_Lower" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"],
            line=dict(color="#585b70", width=1), name="BB Upper"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"],
            line=dict(color="#585b70", width=1), name="BB Lower"), row=1, col=1)

    if "RSI_14" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI_14"],
            line=dict(color="#89dceb", width=2), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#f38ba8", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#a6e3a1", row=2, col=1)
        fig.update_yaxes(range=[0, 100], row=2, col=1)

    if "MACD" in df.columns and "MACD_Signal" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"],
            line=dict(color="#cba6f7", width=2), name="MACD"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"],
            line=dict(color="#f9e2af", width=1.5), name="Signal"), row=3, col=1)
        if "MACD_Hist" in df.columns:
            colors = ["#a6e3a1" if v >= 0 else "#f38ba8" for v in df["MACD_Hist"]]
            fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"],
                marker_color=colors, name="Hist"), row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10),
        height=600,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)

    return fig

def format_timeframe_summary(tf_data, instrument):
    if not tf_data:
        return
    cfg = INSTRUMENTS[instrument]
    d = cfg["decimals"]
    rows = []
    for tf in TF_NAMES:
        if tf not in tf_data or tf_data[tf].empty:
            continue
        df = tf_data[tf]
        last = df.iloc[-1]
        trend, _ = determine_trend(df)
        price = float(last["Close"])
        rsi_val = last.get("RSI_14", np.nan)
        rsi = float(rsi_val) if not pd.isna(rsi_val) else None
        rsi_str = f"{rsi:.1f}" if rsi else "N/A"
        rsi_status = "Overbought" if (rsi and rsi > 70) else ("Oversold" if (rsi and rsi < 30) else "Neutral") if rsi else "N/A"
        rows.append({
            "TF": tf,
            "Trend": trend,
            "Price": price,
            "RSI": rsi_str,
            "Signal": rsi_status,
        })
    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "TF": "TF",
                "Trend": "Trend",
                "Price": st.column_config.NumberColumn("Price", format=f"%.{d}f"),
                "RSI": "RSI",
                "Signal": "Signal",
            },
        )

def render_regime(tf_data):
    if not tf_data or "1H" not in tf_data:
        return
    try:
        regime = detect_regime(tf_data["1H"])
        rec = regime_recommendation(regime)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Regime", regime.get("regime", "N/A").replace("_", " ").title())
        with c2:
            st.metric("Trend Strength", f'{regime.get("trend_strength", 0):.1f}')
        with c3:
            st.metric("Volatility", regime.get("volatility", "N/A"))
        st.info(rec)
    except Exception as e:
        st.caption(f"Regime: {e}")

def render_divergence(tf_data):
    if not tf_data:
        return
    try:
        divs = analyze_all_divergences(tf_data, INSTRUMENTS.get(st.session_state.instrument, {}))
        if divs:
            for div in divs:
                sig = div.get("signal", "neutral")
                emoji = "🟢" if sig == "bullish" else ("🔴" if sig == "bearish" else "⚪")
                st.markdown(f"{emoji} **{div.get('type', 'div')}** on {div.get('timeframe', '?')}: {div.get('description', '')}")
        else:
            st.caption("No divergences detected")
    except Exception as e:
        st.caption(f"Divergence: {e}")

def render_confluence(tf_data):
    try:
        con = build_confluence_summary(tf_data)
        st.markdown(f"```\n{con}\n```")
    except Exception as e:
        st.caption(f"Confluence: {e}")

# ═══════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════

def main():
    # ── Header ──
    st.markdown("<h1 style='text-align: center;'>📈 Trading Chat</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: #a6adc8; font-size: 0.9rem;'>"
        "Multi‑timeframe · Position · Swing · Day Trade · Scalp · Ichimoku · SMC · AI"
        "</p>",
        unsafe_allow_html=True,
    )

    # ── Instrument row ──
    ci1, ci2, ci3 = st.columns([3, 1, 1])
    with ci1:
        instr = st.selectbox(
            "Instrument",
            options=INSTRUMENT_KEYS,
            format_func=lambda x: f"{x.upper()} — {INSTRUMENTS[x]['display_name']}",
            key="instrument",
        )
    with ci3:
        st.markdown("<br>", unsafe_allow_html=True)
        refresh = st.button("🔄 Refresh", type="primary")

    # ── Fetch data ──
    with st.spinner("📡 Fetching market data..."):
        tf_data = cached_fetch_timeframes(instr)
        st.session_state.tf_data = tf_data

    if not tf_data:
        st.error("❌ Failed to fetch data. Check internet connection.")
        return

    cfg = INSTRUMENTS[instr]
    d = cfg["decimals"]
    spot = get_latest_price(tf_data)

    # ── Live price ──
    st.markdown(
        f"<div style='text-align: center; padding: 0.7rem; background: #1e1e2e; "
        f"border-radius: 14px; margin-bottom: 1rem; border: 1px solid #313244;'>"
        f"<span style='font-size: 2rem; font-weight: 700;'>{fmt_price_st(spot, d)}</span>"
        f"<span style='color: #a6adc8; margin-left: 0.5rem;'>{cfg['display_name']}</span>"
        f"<span style='color: #585b70; margin-left: 0.5rem; font-size: 0.8rem;'>"
        f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Strategy buttons ──
    st.markdown("### 🎯 Strategy")
    strategies = [
        ("📊 Data", "data"),
        ("📈 Position", "position"),
        ("🔀 Swing", "swing"),
        ("⚡ Day Trade", "daytrade"),
    ]
    strategies2 = [
        ("🎯 Scalp", "scalp"),
        ("🍁 Ichimoku", "ichimoku"),
        ("🤖 SMC", "smc"),
        ("💬 AI Chat", "ai"),
    ]
    for row in [strategies, strategies2]:
        cols = st.columns(len(row))
        for i, (label, key) in enumerate(row):
            with cols[i]:
                if st.button(label, key=f"s_{key}", use_container_width=True):
                    st.session_state.active_strategy = key
                    st.session_state.last_result = ""

    active = st.session_state.active_strategy

    # ═══════════════════════════════════════════
    #  TABS
    # ═══════════════════════════════════════════
    tabs = st.tabs(["📊 Summary", "📈 Charts", "📝 Analysis"])

    # ── TAB 1: Summary ──
    with tabs[0]:
        st.markdown("### 📊 Timeframe Summary")
        format_timeframe_summary(tf_data, instr)

        col_r, col_d = st.columns(2)
        with col_r:
            st.markdown("### 🏁 Regime")
            render_regime(tf_data)
        with col_d:
            st.markdown("### 🔍 Divergence")
            render_divergence(tf_data)

        st.markdown("### 🔗 Confluence")
        render_confluence(tf_data)

        if cfg.get("has_spot"):
            try:
                sp = fetch_spot_price(cfg["spot_url"], instr, cfg.get("symbol", ""))
                if sp:
                    st.metric("Spot (Investing.com)", fmt_price_st(sp, d))
            except:
                pass

    # ── TAB 2: Charts ──
    with tabs[1]:
        sel_tf = st.selectbox("Timeframe", TF_NAMES, index=2)
        if sel_tf in tf_data and not tf_data[sel_tf].empty:
            fig = plot_candlestick(tf_data[sel_tf], sel_tf)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("Not enough data to plot")
        else:
            st.caption(f"No data for {sel_tf}")

    # ── TAB 3: Analysis ──
    with tabs[2]:
        if active == "ai":
            st.markdown("### 💬 Chat với AI")
            user_q = st.text_area(
                "Hỏi về thị trường (e.g., xu hướng, entry, SL/TP...)",
                placeholder="Phân tích XAUUSD hôm nay...",
                height=100,
            )
            if st.button("🚀 Gửi câu hỏi", type="primary", use_container_width=True):
                if user_q.strip():
                    with st.spinner("🤖 Đang phân tích với DeepSeek..."):
                        try:
                            report = format_report(tf_data, cfg)
                            # Ưu tiên Streamlit secrets, fallback os.environ
                            api_key = st.secrets.get("DEEPSEEK_API_KEY", "") or os.environ.get("DEEPSEEK_API_KEY", "")
                            if api_key:
                                result = call_deepseek(report, user_q, api_key, cfg)
                                st.session_state.last_result = result or "⚠️ AI không phản hồi."
                            else:
                                st.session_state.last_result = "⚠️ Chưa set DEEPSEEK_API_KEY — tạo .streamlit/secrets.toml"
                        except Exception as e:
                            st.session_state.last_result = f"❌ Lỗi: {e}"
                else:
                    st.warning("Nhập câu hỏi trước khi gửi.")
        else:
            strategy_fn_map = {
                "data": (None, "data"),
                "position": (run_position_analysis, "position"),
                "swing": (run_swing_analysis, "swing"),
                "daytrade": (run_daytrade_analysis, "daytrade"),
                "scalp": (run_scalp_analysis, "scalp"),
                "ichimoku": (run_ichimoku_analysis, "ichimoku"),
                "smc": (None, "smc"),
            }
            fn, sname = strategy_fn_map.get(active, (None, "data"))

            if st.button(f"🚀 Run {sname.title()} Analysis", type="primary", use_container_width=True):
                with st.spinner(f"🔬 Analyzing {sname}..."):
                    try:
                        if fn:
                            result = fn(instr)
                        elif sname == "data":
                            result = cached_fetch_data(instr)
                        elif sname == "smc":
                            smc_data = analyze_all_smc(tf_data)
                            result = format_smc_footer(smc_data)
                        else:
                            result = "No analysis function."
                        st.session_state.last_result = result
                    except Exception as e:
                        st.session_state.last_result = f"❌ Lỗi: {e}"

        # Display result
        if st.session_state.last_result:
            st.markdown("### 📋 Result")
            with st.container():
                text = st.session_state.last_result
                if text.startswith("#") or text.startswith("|") or "**" in text:
                    st.markdown(text)
                else:
                    st.text(text)

    # ── Footer ──
    st.markdown("---")
    st.caption(
        f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC &nbsp;·&nbsp; "
        f"Data: Yahoo Finance &nbsp;·&nbsp; AI: DeepSeek"
    )


if __name__ == "__main__":
    main()
