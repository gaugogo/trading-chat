#!/usr/bin/env python3
"""
Multi-Timeframe Analysis Tool with DeepSeek AI Integration

Refactored: shared functions now live in core.py
analysis.py re-exports from core.py for backward compatibility
and keeps only unique functions: format_report, call_deepseek, main

Usage:
  python analysis.py                              # default: XAUUSD
  python analysis.py --instrument btc
  python analysis.py -i gbp --no-ai
"""

import os
import re
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, Any, List
from urllib.parse import quote as url_quote

import numpy as np
import pandas as pd
import requests
from rich.console import Console
from rich.markdown import Markdown

from instruments import INSTRUMENTS
from smc import analyze_all_smc, format_smc_footer, smc_signal_compact
from price_action import analyze_price_action, format_pa_summary, format_pa_compact

# ─── RE-EXPORT FROM core.py ───

from core import (
    CACHE_DIR,
    TIMEFRAMES,
    TF_WEIGHTS,
    PRICE_COLS,
    TF_ORDER,
    fmt_price,
    resample_manual,
    fetch_chart_data,
    fetch_all_timeframes,
    fetch_spot_price,
    adjust_to_spot,
    calculate_atr,
    calculate_indicators,
    determine_trend,
    build_confluence_summary,
)

console = Console()

# ─── GLOBALS ───
DEEPSEEK_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL: str = "https://api.deepseek.com/chat/completions"

_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line.startswith("DEEPSEEK_API_KEY=") and not DEEPSEEK_API_KEY:
                DEEPSEEK_API_KEY = _line.split("=", 1)[1].strip().strip("\"'")


# ─── REPORT (unique to analysis.py) ───

def build_raw_data_summary(tf_data: Dict[str, pd.DataFrame], cfg: Dict[str, Any]) -> str:
    """Build raw data dump for DeepSeek AI — no local analysis, just indicators & levels."""
    d = cfg["decimals"]
    lines: List[str] = []
    lines.append(f"=== RAW MARKET DATA: {cfg['display_name']} ===")
    lines.append(f"Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("")

    for tf_name in TF_ORDER:
        if tf_name not in tf_data or tf_data[tf_name].empty or len(tf_data[tf_name]) < 5:
            continue
        df = tf_data[tf_name]
        last = df.iloc[-1]
        lines.append(f"--- {tf_name} ({len(df)} candles) ---")
        lines.append(f"  O={fmt_price(last['Open'], d)} H={fmt_price(last['High'], d)} L={fmt_price(last['Low'], d)} C={fmt_price(last['Close'], d)}")

        for col, label in [('SMA_20','SMA20'),('SMA_50','SMA50'),('SMA_200','SMA200'),('EMA_9','EMA9'),('EMA_21','EMA21')]:
            val = last.get(col, np.nan)
            if not pd.isna(val):
                lines.append(f"  {label}: {fmt_price(val, d)}")

        for col, label in [('RSI_14','RSI14'),('MACD','MACD'),('MACD_Signal','MACD_Signal'),('MACD_Hist','MACD_Hist')]:
            val = last.get(col, np.nan)
            if not pd.isna(val):
                lines.append(f"  {label}: {float(val):.2f}")

        for col, label in [('BB_Upper','BB_Upper'),('BB_Middle','BB_Mid'),('BB_Lower','BB_Lower'),('BB_Width','BB_Width')]:
            val = last.get(col, np.nan)
            if not pd.isna(val):
                if 'Width' in label:
                    lines.append(f"  {label}: {float(val)*100:.2f}%")
                else:
                    lines.append(f"  {label}: {fmt_price(val, d)}")

        atr = last.get('ATR', np.nan)
        if not pd.isna(atr):
            lines.append(f"  ATR14: {fmt_price(atr, d)}")

        vol = last.get('Volume', np.nan)
        vol_sma = last.get('Volume_SMA_20', np.nan)
        if not pd.isna(vol) and not pd.isna(vol_sma) and vol_sma > 0:
            lines.append(f"  Volume: {vol:,.0f} (vs SMA20: {vol_sma:,.0f}, ratio: {vol/vol_sma:.2f}x)")

        if len(df) >= 20:
            r20 = df['High'].rolling(20).max().iloc[-1]
            s20 = df['Low'].rolling(20).min().iloc[-1]
            lines.append(f"  Resistance(20): {fmt_price(r20, d)}  Support(20): {fmt_price(s20, d)}")
        if len(df) >= 50:
            r50 = df['High'].rolling(50).max().iloc[-1]
            s50 = df['Low'].rolling(50).min().iloc[-1]
            lines.append(f"  Resistance(50): {fmt_price(r50, d)}  Support(50): {fmt_price(s50, d)}")
        lines.append("")

    return "\n".join(lines)


def format_report(tf_data: Dict[str, pd.DataFrame], cfg: Dict[str, Any], spot_price: Optional[float] = None) -> str:
    """Build detailed multi-timeframe analysis report."""
    d = cfg["decimals"]
    if spot_price is None and cfg["has_spot"]:
        spot_price = fetch_spot_price(
            cfg.get("spot_url", ""),
            instrument_id=cfg.get("id", "xau"),
            symbol=cfg.get("symbol", ""),
        )
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append(f"  {cfg['display_name']} \u2014 MULTI-TIMEFRAME ANALYSIS")
    lines.append(f"  Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    if spot_price is not None:
        ref_price = next(
            (d.iloc[-1]['Close'] for d in tf_data.values() if not d.empty), None
        )
        if ref_price is not None:
            source_label = cfg.get("spot_label", "Spot")
            diff = spot_price - ref_price
            lines.append(f"  {source_label}: {fmt_price(spot_price, d)}  |  Yahoo Finance: {fmt_price(ref_price, d)}  (diff: {fmt_price(diff, d)})")
        else:
            lines.append(f"  {cfg.get('spot_label', 'Spot')}: {fmt_price(spot_price, d)}  (source: Investing.com)")
    lines.append("=" * 72)
    lines.append("")

    lines.append("\u3010TREND SUMMARY\u3011")
    lines.append(f"{'TF':<8} {'Trend':<10} {'Price':<14} {'RSI':<10} {'MACD':<12} {'Signal'}")
    lines.append("-" * 70)
    for tf_name in TF_ORDER:
        if tf_name not in tf_data or tf_data[tf_name].empty:
            continue
        df = tf_data[tf_name]
        last = df.iloc[-1]
        trend, _ = determine_trend(df)
        close_str = fmt_price(last['Close'], d)
        rsi_str = f"{last['RSI_14']:.1f}" if not pd.isna(last.get('RSI_14', np.nan)) else "N/A"
        macd_str = f"{last['MACD']:.1f}" if not pd.isna(last.get('MACD', np.nan)) else "N/A"
        if trend == "UP":
            sig = "\U0001f7e2 BUY"
        elif trend == "DOWN":
            sig = "\U0001f534 SELL"
        else:
            sig = "\U0001f7e1 WAIT"
        lines.append(f"{tf_name:<8} {trend:<10} {close_str:<14} {rsi_str:<10} {macd_str:<12} {sig}")
    lines.append("")

    for tf_name in TF_ORDER:
        if tf_name not in tf_data or tf_data[tf_name].empty or len(tf_data[tf_name]) < 5:
            continue
        df = tf_data[tf_name]
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last

        lines.append(f"\n\u3010{tf_name}\u3011({len(df)} candles)")
        lines.append(
            f"  Price:    {fmt_price(last['Close'], d)}  "
            f"(O:{fmt_price(last['Open'], d)} H:{fmt_price(last['High'], d)} L:{fmt_price(last['Low'], d)})"
        )
        change = last['Close'] - prev['Close']
        change_pct = change / prev['Close'] * 100
        lines.append(f"  Change:   {fmt_price(change, d)} ({change_pct:+.2f}%)")

        vol = last.get('Volume', np.nan)
        vol_sma = last.get('Volume_SMA_20', np.nan)
        if not pd.isna(vol) and not pd.isna(vol_sma) and vol_sma > 0:
            vol_ratio = vol / vol_sma
            if vol_ratio > 2.0:
                vol_note = "\u26a0\ufe0f Very high (2x+ avg)"
            elif vol_ratio > 1.5:
                vol_note = "High (1.5x avg)"
            elif vol_ratio < 0.5:
                vol_note = "Low (0.5x avg)"
            else:
                vol_note = "Normal"
            lines.append(f"  Volume:   {vol:,.0f} (SMA20: {vol_sma:,.0f}, {vol_ratio:.2f}x \u2014 {vol_note})")

        sma_parts = []
        for name in ['SMA_20', 'SMA_50', 'SMA_200']:
            val = last.get(name, np.nan)
            if not pd.isna(val):
                sma_parts.append(f"{name.replace('_','')}: {fmt_price(val, d)}")
        if sma_parts:
            lines.append(f"  SMAs:     {' | '.join(sma_parts)}")

        ema_parts = []
        for name in ['EMA_9', 'EMA_21']:
            val = last.get(name, np.nan)
            if not pd.isna(val):
                ema_parts.append(f"{name.replace('_','')}: {fmt_price(val, d)}")
        if ema_parts:
            lines.append(f"  EMAs:     {' | '.join(ema_parts)}")

        pos_parts = []
        for name in ['SMA_20', 'SMA_50', 'SMA_200']:
            val = last.get(name, np.nan)
            if not pd.isna(val):
                pos_parts.append(
                    f"{name.replace('_','')} ABOVE \u2713" if last['Close'] > val
                    else f"{name.replace('_','')} BELOW \u2717"
                )
        if pos_parts:
            lines.append(f"  Position: {' | '.join(pos_parts)}")

        rsi = last.get('RSI_14', np.nan)
        if not pd.isna(rsi):
            if rsi > 70:
                rsi_note = "\u26a0\ufe0f Overbought"
            elif rsi > 60:
                rsi_note = "Strong bullish"
            elif rsi > 50:
                rsi_note = "Bullish zone"
            elif rsi > 40:
                rsi_note = "Bearish zone"
            elif rsi > 30:
                rsi_note = "Weak bearish"
            else:
                rsi_note = "\u26a0\ufe0f Oversold"
            lines.append(f"  RSI(14):  {rsi:.2f} \u2014 {rsi_note}")

        macd_v = last.get('MACD', np.nan)
        macd_s = last.get('MACD_Signal', np.nan)
        if not pd.isna(macd_v) and not pd.isna(macd_s):
            macd_note = "Bullish (above signal)" if macd_v > macd_s else "Bearish (below signal)"
            lines.append(f"  MACD:     {macd_v:.2f} / Signal: {macd_s:.2f} \u2014 {macd_note}")
            hist = last.get('MACD_Hist', np.nan)
            if not pd.isna(hist):
                lines.append(f"  Hist:     {hist:.2f} ({'increasing' if hist > 0 else 'decreasing'})")

        bb_mid = last.get('BB_Middle', np.nan)
        bb_up = last.get('BB_Upper', np.nan)
        bb_low = last.get('BB_Lower', np.nan)
        if not pd.isna(bb_mid):
            bb_pos = (last['Close'] - bb_low) / (bb_up - bb_low) * 100 if (bb_up - bb_low) > 0 else 50
            bb_w = last.get('BB_Width', np.nan)
            lines.append(f"  BB(20,2): Upper:{fmt_price(bb_up, d)} | Mid:{fmt_price(bb_mid, d)} | Lower:{fmt_price(bb_low, d)}")
            bbp_note = (
                "touch upper \u26a0\ufe0f" if bb_pos >= 100 else
                "near upper" if bb_pos >= 80 else
                "middle" if bb_pos >= 20 else
                "near lower" if bb_pos > 0 else
                "touch lower \u26a0\ufe0f"
            )
            lines.append(f"  BB %B:    {bb_pos:.1f}% ({bbp_note})")
            if not pd.isna(bb_w):
                lines.append(f"  BB Width: {bb_w*100:.2f}% ({'narrowing \u26a1' if bb_w < 0.05 else 'wide'} )")

        if len(df) >= 20:
            res = df['High'].rolling(20).max().iloc[-1]
            sup = df['Low'].rolling(20).min().iloc[-1]
            lines.append(f"  R(20):    {fmt_price(res, d)} | S(20): {fmt_price(sup, d)}")
        if len(df) >= 50:
            res50 = df['High'].rolling(50).max().iloc[-1]
            sup50 = df['Low'].rolling(50).min().iloc[-1]
            lines.append(f"  R(50):    {fmt_price(res50, d)} | S(50): {fmt_price(sup50, d)}")

        atr = last.get('ATR', np.nan)
        if not pd.isna(atr):
            lines.append(f"  ATR(14):  {fmt_price(atr, d)}")

        lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)


# ─── DEEPSEEK API INTEGRATION ───

def call_deepseek(report_text: str, user_question: str, api_key: str, cfg: Dict[str, Any]) -> Optional[str]:
    """Call DeepSeek API with analysis report and user question.

    Uses educational prompt templates if available, otherwise falls back
    to the legacy simple prompt.
    """
    system_prompt = (
        f"Bạn là chuyên gia phân tích kỹ thuật {cfg['prompt_instrument']}. "
        "Nhận dữ liệu thô, tự phân tích toàn bộ. "
        "Trả lời TIẾNG VIỆT, dùng số liệu cụ thể. "
        "Bias → Entry/SL/TP → Rủi ro → Bài học."
    )

    # Use educational prompt template if available
    try:
        from education.prompt_templates import build_system_prompt, build_user_prompt
        system_prompt = build_system_prompt(cfg)
        user_content = build_user_prompt(report_text, cfg, user_question)
    except ImportError:
        user_content = f"Dữ liệu {cfg['prompt_instrument']}:\n\n{report_text}\n\n{user_question}"

    payload = {
        "model": cfg["deepseek_model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.3,
        "max_tokens": 4096,
        "stream": False,
    }

    if cfg["deepseek_thinking"]:
        payload["thinking"] = {"type": "enabled"}
        payload["reasoning_effort"] = "high"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    print("  Calling DeepSeek API... ", end="", flush=True)
    try:
        resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        result = resp.json()
        content = result["choices"][0]["message"]["content"]
        print("\u2705 Done!")
        return content
    except requests.exceptions.Timeout:
        print("\u274c Timeout (API took > 120s)")
        return None
    except requests.exceptions.RequestException as e:
        print(f"\u274c API Error: {e}")
        if e.response:
            print(f"  Response: {e.response.text[:500]}")
        return None
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"\u274c Parse error: {e}")
        return None


# ─── MAIN ───

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Timeframe Technical Analysis Tool")
    parser.add_argument("--instrument", "-i", default="xau", choices=list(INSTRUMENTS.keys()),
                        help="Trading instrument (default: xau)")
    parser.add_argument("--symbol", "-s", help="Yahoo Finance symbol (overrides instrument default)")
    parser.add_argument("--no-ai", action="store_true", help="Skip DeepSeek AI analysis")
    parser.add_argument("--no-cache", action="store_true", help="Bypass local cache")
    parser.add_argument("--smc", action="store_true", help="Include SMC (Smart Money Concepts) analysis")
    parser.add_argument("--pa", action="store_true", help="Include Price Action analysis (candlestick patterns, S/R, structure)")
    parser.add_argument("--output", "-o", default=".", help="Output directory for reports")
    args = parser.parse_args()

    cfg = dict(INSTRUMENTS[args.instrument])
    if args.symbol:
        cfg["symbol"] = args.symbol
        cfg["symbol_encoded"] = url_quote(args.symbol, safe='')
        cfg["display_name"] = f"{args.symbol} ({cfg.get('prompt_instrument', args.symbol)})"

    print(f"Fetching {cfg['display_name']}\u2026")
    print(f"  Source: Yahoo Finance  |  Symbol: {cfg['symbol']}")
    print()

    tf_data = fetch_all_timeframes(cfg, use_cache=not args.no_cache)
    for tf_name in TF_ORDER:
        if tf_name in tf_data:
            df = tf_data[tf_name]
            print(f"  {tf_name:5s}     {len(df):>4d} candles @ {fmt_price(df['Close'].iloc[-1], cfg['decimals'])}")
        else:
            print(f"  {tf_name:5s}     \u274c no data")

    print()
    if not tf_data:
        print("No data retrieved. Check symbol or internet connection.")
        return

    # Build raw data for AI (no local analysis displayed)
    raw_report = format_report(tf_data, cfg)
    raw_data = build_raw_data_summary(tf_data, cfg)

    if args.smc and cfg["has_smc"]:
        print("\n  Running SMC analysis...")
        smc_data = analyze_all_smc(tf_data)
        raw_report += format_smc_footer(smc_data)
    elif args.smc and not cfg["has_smc"]:
        print("  \u26a0\ufe0f SMC analysis not available for this instrument (xau only).")

    if args.pa:
        print("\n  Running Price Action analysis...")
        pa_data = analyze_price_action(tf_data, cfg)
        raw_report += "\n" + format_pa_summary(pa_data, cfg["decimals"])
        print("  ✅ Done!")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = cfg["file_prefix"]

    if args.no_ai or not DEEPSEEK_API_KEY:
        if not DEEPSEEK_API_KEY:
            print("\n\u2139\ufe0f  To auto-analyze with DeepSeek AI, set DEEPSEEK_API_KEY:")
            print("   export DEEPSEEK_API_KEY=sk-your-key-here")
            print("   Or create a .env file next to this script with:")
            print('   DEEPSEEK_API_KEY="sk-your-key-here"')
            print()
        print("\u2500\u2500\u2500 Raw Data for DeepSeek \u2500\u2500\u2500")
        print(raw_data)
        print("\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
        if args.no_ai:
            print("Skipped AI analysis (--no-ai).")
        else:
            print("Copy the data above and paste into DeepSeek chat.")
        return

    print("\n" + "=" * 72)
    print("  \U0001f916 DEEPSEEK AI ANALYSIS")
    print("=" * 72)

    default_q = (
        "Based on this multi-timeframe data, give me a detailed trading plan:\n"
        "1. What is the overall market bias?\n"
        "2. What are the key support and resistance levels?\n"
        "3. Where should I enter, place stop loss, and take profit?\n"
        "4. What is the recommended risk/reward ratio?\n"
        "5. Any divergence or warning signs?"
    )

    print("\nEnter your question for DeepSeek (or press Enter for default trading plan):")
    try:
        user_input = input(">>> ").strip()
    except (EOFError, KeyboardInterrupt):
        user_input = ""

    question = user_input if user_input else default_q

    print()
    ai_response = call_deepseek(raw_report, question, DEEPSEEK_API_KEY, cfg)

    if ai_response:
        print("\n" + "=" * 72)
        print("  \U0001f916 DEEPSEEK ANALYSIS RESULT")
        print("=" * 72)
        print()
        console.print(Markdown(ai_response))
        print()
        print("=" * 72)

        ai_filename = out_dir / f"{prefix}_deepseek_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt"
        with open(ai_filename, "w") as f:
            f.write(ai_response)
        print(f"\U0001f4c4 AI response saved to: {ai_filename}")
    else:
        print("\n\u26a0\ufe0f  Could not get AI analysis. Data has been collected — you can paste it into DeepSeek chat manually.")


if __name__ == "__main__":
    main()
