#!/usr/bin/env python3
"""
Data fetcher for Pi Agent Trading Tools.
Provides raw multi-timeframe market data (OHLCV + indicators)
for the Pi Agent + DeepSeek to analyze.

Supports: XAUUSD, BTC/USD, GBP/USD
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from typing import Dict, List

from core import (
    fetch_all_timeframes,
    fetch_spot_price,
    determine_trend,
    build_confluence_summary,
    adjust_to_spot,
    fmt_price,
    TF_ORDER,
)
from instruments import INSTRUMENTS


# ─── DATA OUTPUT FORMATTER ───

# fmt_price imported from core (handles NaN, None, Series)


def fmt_compact_data(tf_data: Dict[str, pd.DataFrame], cfg: dict) -> str:
    """Return raw data in compact format optimized for AI reasoning."""
    lines: List[str] = []
    d = cfg["decimals"]
    name = cfg["display_name"]

    # Header
    lines.append(f"# {name}")
    lines.append(f"Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("")

    # ── Summary per timeframe ──
    lines.append("## SUMMARY")
    lines.append("TF     |Trend    |Price         |RSI   |MACD/H   |SMA20        |SMA50        |BB_U         |BB_L         |ATR        ")
    lines.append("-------|---------|--------------|------|---------|-------------|-------------|-------------|-------------|-----------")

    for tf_name in ["Daily", "4H", "1H", "15m", "5m"]:
        if tf_name not in tf_data or tf_data[tf_name].empty:
            continue
        df = tf_data[tf_name]
        last = df.iloc[-1]
        trend, _ = determine_trend(df)

        close_s = fmt_price(last["Close"], d)
        rsi_s = f"{last['RSI_14']:.1f}" if not pd.isna(last.get("RSI_14", np.nan)) else "N/A"

        macd_v = last.get("MACD", np.nan)
        macd_s = last.get("MACD_Signal", np.nan)
        if not pd.isna(macd_v) and not pd.isna(macd_s):
            macd_s = f"{macd_v:.1f}/{macd_s:.1f}"
        else:
            macd_s = "N/A"

        sma20_s = fmt_price(last.get("SMA_20", np.nan), d)
        sma50_s = fmt_price(last.get("SMA_50", np.nan), d)
        bb_u_s = fmt_price(last.get("BB_Upper", np.nan), d)
        bb_l_s = fmt_price(last.get("BB_Lower", np.nan), d)
        atr_s = fmt_price(last.get("ATR", np.nan), d)

        lines.append(
            f"{tf_name:<7}|{trend:<9}|{close_s:<14}|{rsi_s:<6}|{macd_s:<9}|"
            f"{sma20_s:<13}|{sma50_s:<13}|{bb_u_s:<13}|{bb_l_s:<13}|{atr_s:<11}"
        )

    lines.append("")

    # ── Confluence ──
    lines.append("## CONFLUENCE")
    lines.append(build_confluence_summary(tf_data))
    lines.append("")

    # ── Raw OHLCV tails (last 15 candles per TF) ──
    lines.append("## RAW CANDLES")
    for tf_name in ["Daily", "4H", "1H", "15m", "5m"]:
        if tf_name not in tf_data or tf_data[tf_name].empty:
            continue
        df = tf_data[tf_name]
        lines.append(f"\n### {tf_name} ({len(df)} candles, last 15)")
        lines.append("Date                 |Open          |High          |Low           |Close         |Volume   |RSI   |MACD   ")
        lines.append("---------------------|--------------|--------------|--------------|--------------|---------|------|--------")

        tail = df.tail(15)
        for idx, row in tail.iterrows():
            dt = idx.strftime("%Y-%m-%d %H:%M")
            o = fmt_price(row["Close"], d)  # approximate; use Open
            o = fmt_price(row["Open"], d)
            h = fmt_price(row["High"], d)
            l = fmt_price(row["Low"], d)
            c = fmt_price(row["Close"], d)
            vol = f"{row['Volume']:.0f}" if not pd.isna(row.get("Volume", np.nan)) else "N/A"
            rsi = f"{row['RSI_14']:.1f}" if not pd.isna(row.get("RSI_14", np.nan)) else "N/A"
            macd = f"{row['MACD']:.1f}" if not pd.isna(row.get("MACD", np.nan)) else "N/A"
            lines.append(f"{dt:<21}|{o:<14}|{h:<14}|{l:<14}|{c:<14}|{vol:<9}|{rsi:<6}|{macd:<8}")

    lines.append("")

    # ── Key levels ──
    lines.append("## KEY LEVELS")
    for tf_name in ["Daily", "4H", "1H"]:
        if tf_name not in tf_data or tf_data[tf_name].empty:
            continue
        df = tf_data[tf_name]
        if len(df) >= 20:
            r20 = df["High"].rolling(20).max().iloc[-1]
            s20 = df["Low"].rolling(20).min().iloc[-1]
            lines.append(f"  {tf_name} R(20): {fmt_price(r20, d)}  S(20): {fmt_price(s20, d)}")
        if len(df) >= 50:
            r50 = df["High"].rolling(50).max().iloc[-1]
            s50 = df["Low"].rolling(50).min().iloc[-1]
            lines.append(f"  {tf_name} R(50): {fmt_price(r50, d)}  S(50): {fmt_price(s50, d)}")

    return "\n".join(lines)





# ─── PUBLIC API ───


def fetch_data(instrument: str = "xau", no_cache: bool = False) -> str:
    """
    Fetch raw multi-timeframe market data for the Pi Agent.
    Returns compact text with OHLCV + indicators — no AI processing.
    """
    if instrument not in INSTRUMENTS:
        return f"Unknown instrument: {instrument}. Choose from: {', '.join(INSTRUMENTS.keys())}"

    cfg = INSTRUMENTS[instrument]
    tf_data = fetch_all_timeframes(cfg, use_cache=not no_cache)

    if not tf_data:
        return "No data retrieved. Check internet connection."

    tf_data = adjust_to_spot(tf_data, cfg)

    return fmt_compact_data(tf_data, cfg)


if __name__ == "__main__":
    import sys
    instr = sys.argv[1] if len(sys.argv) > 1 else "xau"
    print(fetch_data(instr))
