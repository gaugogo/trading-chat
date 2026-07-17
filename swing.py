#!/usr/bin/env python3
"""
Swing Trading Analysis Module
Focus: Daily, 4H (bias), 1H (entry)
Hold: 1-5 ngày
Strategy: Trend-following, pullback entries, Fibonacci, key S/R zones
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from core import (
    adjust_to_spot,
    fetch_all_timeframes,
    fetch_spot_price,
    calculate_indicators,
    determine_trend,
    fmt_price,
    TF_ORDER,
)
from instruments import INSTRUMENTS

# ─── Swing config ─────────────────────────────────────────────────────

BIAS_TIMEFRAMES = ["Daily", "4H"]      # TF xác định xu hướng chính
ENTRY_TIMEFRAME = "1H"                  # TF tìm entry
CONFIRM_TIMEFRAME = "15m"              # TF xác nhận entry

SWING_ATR_MULTIPLIER_SL = 2.0          # SL = 2x ATR
SWING_ATR_MULTIPLIER_TP1 = 2.5         # TP1 = 2.5x ATR
SWING_ATR_MULTIPLIER_TP2 = 5.0         # TP2 = 5x ATR


# ─── Fibonacci levels ──────────────────────────────────────────────────

def calculate_fib_levels(df: pd.DataFrame, lookback: int = 50) -> Optional[Dict[str, float]]:
    """Tính Fibonacci retracement từ swing high/low gần nhất."""
    if df.empty or len(df) < lookback:
        return None

    recent = df.iloc[-lookback:]
    swing_high = float(recent["High"].max())
    swing_low = float(recent["Low"].min())
    price = float(recent["Close"].iloc[-1])
    range_val = swing_high - swing_low

    if range_val <= 0:
        return None

    # Xác định xu hướng để biết chiều Fibonacci
    trend, _ = determine_trend(df)
    if trend == "UP":
        # Pullback trong uptrend: Fib từ swing_low lên swing_high
        levels = {
            "0.0": round(swing_low, 2),
            "0.236": round(swing_high - range_val * 0.236, 2),
            "0.382": round(swing_high - range_val * 0.382, 2),
            "0.5": round(swing_high - range_val * 0.5, 2),
            "0.618": round(swing_high - range_val * 0.618, 2),
            "0.786": round(swing_high - range_val * 0.786, 2),
            "1.0": round(swing_high, 2),
            "1.272": round(swing_high + range_val * 0.272, 2),
            "1.618": round(swing_high + range_val * 0.618, 2),
        }
    else:
        # Bounce trong downtrend: Fib từ swing_high xuống swing_low
        levels = {
            "0.0": round(swing_high, 2),
            "0.236": round(swing_low + range_val * 0.236, 2),
            "0.382": round(swing_low + range_val * 0.382, 2),
            "0.5": round(swing_low + range_val * 0.5, 2),
            "0.618": round(swing_low + range_val * 0.618, 2),
            "0.786": round(swing_low + range_val * 0.786, 2),
            "1.0": round(swing_low, 2),
            "1.272": round(swing_low - range_val * 0.272, 2),
            "1.618": round(swing_low - range_val * 0.618, 2),
        }

    levels["swing_high"] = round(swing_high, 2)
    levels["swing_low"] = round(swing_low, 2)
    levels["range"] = round(range_val, 2)
    levels["trend"] = trend
    levels["price"] = round(price, 2)

    # Nơi giá đang ở trong Fib
    if trend == "UP":
        levels["price_in_fib"] = "above 1.0" if price > swing_high else f"at {((swing_high - price) / range_val * 100):.0f}% retrace"
    else:
        levels["price_in_fib"] = "below 1.0" if price < swing_low else f"at {((price - swing_low) / range_val * 100):.0f}% retrace"

    return levels


# ─── Key levels (multiple timeframe S/R) ───────────────────────────────

def find_key_levels(tf_data: Dict[str, pd.DataFrame], decimals: int) -> Dict[str, Any]:
    """Tìm support/resistance quan trọng từ Daily & 4H."""
    levels: Dict[str, List[float]] = {"resistance": [], "support": []}

    for tf_name in ["Daily", "4H"]:
        df = tf_data.get(tf_name)
        if df is None or df.empty or len(df) < 30:
            continue

        # Rolling highs/lows
        for window in [20, 50]:
            if len(df) >= window:
                r = float(df["High"].rolling(window).max().iloc[-1])
                s = float(df["Low"].rolling(window).min().iloc[-1])
                if r not in levels["resistance"]:
                    levels["resistance"].append(r)
                if s not in levels["support"]:
                    levels["support"].append(s)

        # SMA levels
        for sma_name in ["SMA_20", "SMA_50"]:
            val = df[sma_name].iloc[-1] if sma_name in df.columns else None
            if val is not None and not pd.isna(val):
                val = float(val)
                last_close = float(df["Close"].iloc[-1])
                if val > last_close:
                    if val not in levels["resistance"]:
                        levels["resistance"].append(val)
                else:
                    if val not in levels["support"]:
                        levels["support"].append(val)

    # Sort and deduplicate with tolerance
    levels["resistance"] = sorted(set(round(r, decimals) for r in levels["resistance"]), reverse=True)
    levels["support"] = sorted(set(round(s, decimals) for s in levels["support"]))

    return levels


# ─── Swing signal scoring ──────────────────────────────────────────────

def swing_score(tf_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Tính điểm swing dựa trên multi-TF confluence + technicals."""
    details: List[str] = []
    score = 0.0

    # 1. Daily trend (weight: 3)
    daily = tf_data.get("Daily")
    if daily is not None and not daily.empty:
        d_trend, d_score = determine_trend(daily)
        score += d_score * 1.5
        details.append(f"Daily: {d_trend}({d_score:+d}) → {d_score * 1.5:+.1f}")

        last = daily.iloc[-1]
        # SMA alignment
        sma20 = last.get("SMA_20", np.nan)
        sma50 = last.get("SMA_50", np.nan)
        if not pd.isna(sma20) and not pd.isna(sma50):
            if sma20 > sma50:
                score += 1
                details.append(f"Daily SMA20 > SMA50 (golden cross) [+1]")
            elif sma20 < sma50:
                score -= 1
                details.append(f"Daily SMA20 < SMA50 (death cross) [-1]")

        # RSI divergence potential
        rsi = last.get("RSI_14", np.nan)
        if not pd.isna(rsi):
            if d_trend == "UP" and 40 <= rsi <= 50:
                score += 0.5
                details.append(f"Daily RSI={rsi:.1f} in dip zone (bullish) [+0.5]")
            elif d_trend == "DOWN" and 50 <= rsi <= 60:
                score -= 0.5
                details.append(f"Daily RSI={rsi:.1f} in rally zone (bearish) [-0.5]")

    # 2. 4H trend (weight: 2)
    h4 = tf_data.get("4H")
    if h4 is not None and not h4.empty:
        h4_trend, h4_score = determine_trend(h4)
        score += h4_score * 1.0
        details.append(f"4H: {h4_trend}({h4_score:+d}) → {h4_score * 1.0:+.1f}")

        last_h4 = h4.iloc[-1]
        # MACD trend
        macd = last_h4.get("MACD", np.nan)
        macd_sig = last_h4.get("MACD_Signal", np.nan)
        if not pd.isna(macd) and not pd.isna(macd_sig):
            if macd > macd_sig:
                score += 0.5
                details.append(f"4H MACD bullish [+0.5]")
            else:
                score -= 0.5
                details.append(f"4H MACD bearish [-0.5]")

    # 3. 1H entry alignment
    h1 = tf_data.get("1H")
    if h1 is not None and not h1.empty:
        h1_trend, h1_score = determine_trend(h1)
        score += h1_score * 0.7
        details.append(f"1H: {h1_trend}({h1_score:+d}) → {h1_score * 0.7:+.1f}")

        last_h1 = h1.iloc[-1]
        # Bollinger position
        bb_upper = last_h1.get("BB_Upper", np.nan)
        bb_lower = last_h1.get("BB_Lower", np.nan)
        close_h1 = float(last_h1["Close"])
        if not pd.isna(bb_upper) and not pd.isna(bb_lower):
            bb_pos = (close_h1 - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
            if bb_pos < 0.2:
                score += 1
                details.append(f"1H near BB lower (oversold bounce zone) [+1]")
            elif bb_pos > 0.8:
                score -= 1
                details.append(f"1H near BB upper (overbought zone) [-1]")

    # 4. Volume confirmation
    for tf_name in ["Daily", "4H"]:
        df = tf_data.get(tf_name)
        if df is None or df.empty:
            continue
        vol = df["Volume"].iloc[-1]
        vol_sma = df["Volume"].rolling(20).mean().iloc[-1]
        if not pd.isna(vol_sma) and vol_sma > 0:
            vol_ratio = float(vol / vol_sma)
            if vol_ratio > 1.5:
                score += 0.5 if d_trend == "UP" else -0.5
                details.append(f"{tf_name} Volume spike x{vol_ratio:.1f} [{'+0.5' if d_trend=='UP' else '-0.5'}]")

    trend_map = {"UP": 1, "DOWN": -1, "SIDEWAYS": 0}
    bias = "WAIT"
    if score >= 3.5:
        bias = "STRONG BUY"
    elif score >= 1.5:
        bias = "BUY BIAS"
    elif score <= -3.5:
        bias = "STRONG SELL"
    elif score <= -1.5:
        bias = "SELL BIAS"

    return {
        "score": round(score, 1),
        "bias": bias,
        "details": details,
    }


# ─── Entry/Exit zones for swing ────────────────────────────────────────

def swing_entry_zones(
    tf_data: Dict[str, pd.DataFrame],
    bias: str,
    decimals: int,
) -> Optional[Dict[str, Any]]:
    """Tính entry/exit cho swing dựa trên ATR 4H + key levels."""
    h4_df = tf_data.get("4H")
    if h4_df is None or h4_df.empty:
        return None

    last_h4 = h4_df.iloc[-1]
    price = float(last_h4["Close"])

    # ATR 14 trên 4H
    atr_val = float(last_h4.get("ATR", np.nan)) if not pd.isna(last_h4.get("ATR", np.nan)) else float(price * 0.005)

    # Key levels
    levels = find_key_levels(tf_data, decimals)

    if "BUY" in bias:
        # Tìm support gần nhất làm entry zone bottom
        supports_below = [s for s in levels["support"] if s < price]
        key_support = max(supports_below) if supports_below else price - atr_val * 2

        entry_low = max(key_support, price - atr_val * 1.0)
        entry_high = price
        sl = round(entry_low - atr_val * SWING_ATR_MULTIPLIER_SL, decimals)
        tp1 = round(price + atr_val * SWING_ATR_MULTIPLIER_TP1, decimals)
        tp2 = round(price + atr_val * SWING_ATR_MULTIPLIER_TP2, decimals)

        # Tìm resistance cho TP
        resistances_above = [r for r in levels["resistance"] if r > price]
        if resistances_above:
            nearest_r = min(resistances_above)
            if nearest_r < tp1:
                tp1 = round(nearest_r - atr_val * 0.1, decimals)

        risk = price - sl
        rr1 = round((tp1 - price) / risk, 1) if risk > 0 else 0
        rr2 = round((tp2 - price) / risk, 1) if risk > 0 else 0
    else:
        resistances_above = [r for r in levels["resistance"] if r > price]
        key_resistance = min(resistances_above) if resistances_above else price + atr_val * 2

        entry_low = price
        entry_high = min(key_resistance, price + atr_val * 1.0)
        sl = round(entry_high + atr_val * SWING_ATR_MULTIPLIER_SL, decimals)
        tp1 = round(price - atr_val * SWING_ATR_MULTIPLIER_TP1, decimals)
        tp2 = round(price - atr_val * SWING_ATR_MULTIPLIER_TP2, decimals)

        supports_below = [s for s in levels["support"] if s < price]
        if supports_below:
            nearest_s = max(supports_below)
            if nearest_s > tp1:
                tp1 = round(nearest_s + atr_val * 0.1, decimals)

        risk = sl - price
        rr1 = round((price - tp1) / risk, 1) if risk > 0 else 0
        rr2 = round((price - tp2) / risk, 1) if risk > 0 else 0

    return {
        "entry_zone": f"{fmt_price(entry_low, decimals)} - {fmt_price(entry_high, decimals)}",
        "stop_loss": fmt_price(sl, decimals),
        "tp1": fmt_price(tp1, decimals),
        "tp2": fmt_price(tp2, decimals),
        "rr1": rr1,
        "rr2": rr2,
        "atr_4h": fmt_price(atr_val, decimals),
        "key_support": fmt_price(key_support if "BUY" in bias else (max(supports_below) if supports_below else price), decimals),
        "key_resistance": fmt_price(key_resistance if "SELL" in bias else (min(resistances_above) if resistances_above else price), decimals),
    }


# ─── Build swing report ───────────────────────────────────────────────

def build_swing_report(tf_data: Dict[str, pd.DataFrame], cfg: Dict[str, Any]) -> str:
    d = cfg["decimals"]
    name = cfg["display_name"]

    # Check required TFs
    for tf in ["Daily", "4H", "1H"]:
        if tf not in tf_data or tf_data[tf].empty:
            return f"ERROR: Need {tf} data for swing trading."

    # Score
    sig = swing_score(tf_data)
    bias = sig["bias"]

    # Trend summary per TF
    trend_summary = {}
    for tf_name in ["Daily", "4H", "1H", "15m"]:
        df = tf_data.get(tf_name)
        if df is not None and not df.empty:
            trend, score = determine_trend(df)
            trend_summary[tf_name] = (trend, score)

    # Fib levels
    fib = calculate_fib_levels(tf_data.get("4H", pd.DataFrame()), 50)

    # Entry zones
    zones = swing_entry_zones(tf_data, bias, d) if bias != "WAIT" else None

    # Build
    lines: List[str] = []
    lines.append("=" * 64)
    lines.append(f"  📈 SWING TRADING ANALYSIS — {name}")
    lines.append(f"  Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"  Style: Positional (hold 1-5 days)")
    lines.append("=" * 64)
    lines.append("")

    # Multi-TF Overview
    lines.append("【MULTI-TIMEFRAME OVERVIEW】")
    lines.append(f"{'TF':<8} {'Trend':<10} {'Price':<14} {'RSI':<8} {'MACD/Sig':<12} {'SMA20':<12} {'SMA50':<12}")
    lines.append("-" * 72)
    for tf_name in ["Daily", "4H", "1H"]:
        df = tf_data[tf_name]
        last = df.iloc[-1]
        trend, _ = trend_summary[tf_name]
        rsi_s = f"{last['RSI_14']:.1f}" if not pd.isna(last.get("RSI_14", np.nan)) else "N/A"
        macd_s = f"{last['MACD']:.1f}/{last['MACD_Signal']:.1f}" if not pd.isna(last.get("MACD", np.nan)) else "N/A"
        sma20 = fmt_price(last.get("SMA_20", np.nan), d)
        sma50 = fmt_price(last.get("SMA_50", np.nan), d)
        lines.append(f"{tf_name:<8} {trend:<10} {fmt_price(last['Close'], d):<14} {rsi_s:<8} {macd_s:<12} {sma20:<12} {sma50:<12}")
    lines.append("")

    # Score breakdown
    bias_icon = {"STRONG BUY": "🟢", "BUY BIAS": "🟢", "STRONG SELL": "🔴", "SELL BIAS": "🔴"}.get(bias, "🟡")
    lines.append(f"【SWING SIGNAL: {bias_icon} {bias} (score: {sig['score']:+.1f})】")
    lines.append("  Score Breakdown:")
    for detail in sig["details"]:
        lines.append(f"    - {detail}")
    lines.append("")

    # Fibonacci
    if fib:
        lines.append("【FIBONACCI LEVELS — 4H】")
        lines.append(f"  Swing High: {fmt_price(fib['swing_high'], d)}")
        lines.append(f"  Swing Low:  {fmt_price(fib['swing_low'], d)}")
        lines.append(f"  Range:      {fmt_price(fib['range'], d)}")
        lines.append(f"  Current:    {fmt_price(fib['price'], d)} ({fib['price_in_fib']})")
        lines.append(f"  Key Levels:")
        for level in ["0.0", "0.382", "0.5", "0.618", "0.786", "1.0"]:
            marker = " ◀ PRICE" if abs(fib[level] - fib["price"]) < fib["range"] * 0.02 else ""
            lines.append(f"    {level:<6}: {fmt_price(fib[level], d)}{marker}")
        lines.append(f"  Extensions:  1.272: {fmt_price(fib['1.272'], d)}  1.618: {fmt_price(fib['1.618'], d)}")
        lines.append("")

    # Key S/R Levels
    levels = find_key_levels(tf_data, d)
    lines.append("【KEY SUPPORT & RESISTANCE】")
    lines.append(f"  Resistance: {', '.join(fmt_price(r, d) for r in levels['resistance'][:4]) or 'N/A'}")
    lines.append(f"  Support:    {', '.join(fmt_price(s, d) for s in levels['support'][:4]) or 'N/A'}")
    lines.append("")

    # Entry/Exit zones
    if zones:
        lines.append("【ENTRY & EXIT PLAN】")
        lines.append(f"  Entry Zone:    {zones['entry_zone']}")
        lines.append(f"  Stop Loss:     {zones['stop_loss']}")
        lines.append(f"  Take Profit 1: {zones['tp1']} (R:R = 1:{zones['rr1']})")
        lines.append(f"  Take Profit 2: {zones['tp2']} (R:R = 1:{zones['rr2']})")
        lines.append(f"  ATR(4H):       {zones['atr_4h']}")
        lines.append(f"  Key Support:   {zones['key_support']}")
        lines.append(f"  Key Resistance:{zones['key_resistance']}")
        lines.append("")
        lines.append("  📋 Swing Trading Rules:")
        lines.append("  • Entry: Limit order trong vùng entry, không market order")
        lines.append("  • SL: Dựa trên cấu trúc + ATR, không dời SL ngược hướng")
        lines.append("  • TP1: Đóng 50% vị thế, dời SL về hòa vốn")
        lines.append("  • Trailing: Dùng SMA20 hoặc swing low/high để trail phần còn lại")
        lines.append("  • Position size: Risk 1-2% tài khoản mỗi lệnh")
        lines.append("  • Correlation: Kiểm tra DXY, Bond Yields trước khi vào lệnh XAU")
    else:
        lines.append("【NO CLEAR ENTRY】")
        lines.append("  ⏸️  Chưa có setup rõ ràng. Chờ breakout hoặc pullback về vùng giá trị.")
        lines.append("  • Theo dõi break của Daily high/low gần nhất")
        lines.append("  • Chờ giá chạm Fib 0.5-0.618 để tìm entry")

    lines.append("")
    lines.append("=" * 64)
    return "\n".join(lines)


# ─── Quick signal ─────────────────────────────────────────────────────

def build_swing_data_for_ai(tf_data: Dict[str, pd.DataFrame], cfg: Dict[str, Any]) -> str:
    """Build raw data dump for DeepSeek — swing trading context."""
    d = cfg["decimals"]
    name = cfg["display_name"]
    lines: List[str] = []
    lines.append(f"=== SWING TRADING RAW DATA: {name} ===")
    lines.append("")

    for tf_name in ["Daily", "4H", "1H", "15m"]:
        df = tf_data.get(tf_name)
        if df is None or df.empty:
            continue
        last = df.iloc[-1]
        lines.append(f"【{tf_name}】")
        lines.append(f"  O={fmt_price(last['Open'], d)} H={fmt_price(last['High'], d)} L={fmt_price(last['Low'], d)} C={fmt_price(last['Close'], d)}")
        for col, label in [('SMA_20','SMA20'),('SMA_50','SMA50')]:
            val = last.get(col, np.nan)
            if not pd.isna(val):
                lines.append(f"  {label}: {fmt_price(val, d)}")
        for col, label in [('RSI_14','RSI14'),('MACD','MACD'),('MACD_Signal','MACD_Signal')]:
            val = last.get(col, np.nan)
            if not pd.isna(val):
                lines.append(f"  {label}: {float(val):.2f}")
        atr = last.get('ATR', np.nan)
        if not pd.isna(atr):
            lines.append(f"  ATR14: {fmt_price(atr, d)}")
        bb_upper = last.get('BB_Upper', np.nan)
        bb_lower = last.get('BB_Lower', np.nan)
        if not pd.isna(bb_upper) and not pd.isna(bb_lower):
            lines.append(f"  BB_Upper: {fmt_price(bb_upper, d)}  BB_Lower: {fmt_price(bb_lower, d)}")
        lines.append("")

    # Fibonacci
    fib = calculate_fib_levels(tf_data.get("4H", pd.DataFrame()), 50)
    if fib:
        lines.append("【FIBONACCI — 4H】")
        lines.append(f"  Swing High: {fmt_price(fib['swing_high'], d)}")
        lines.append(f"  Swing Low: {fmt_price(fib['swing_low'], d)}")
        lines.append(f"  Range: {fmt_price(fib['range'], d)}")
        lines.append(f"  Current Price: {fmt_price(fib['price'], d)} ({fib['price_in_fib']})")
        for level in ["0.0", "0.382", "0.5", "0.618", "0.786", "1.0", "1.272", "1.618"]:
            lines.append(f"  Fib {level}: {fmt_price(fib[level], d)}")
        lines.append("")

    # Key S/R
    levels = find_key_levels(tf_data, d)
    lines.append("【KEY S/R LEVELS】")
    lines.append(f"  Resistance: {', '.join(fmt_price(r, d) for r in levels['resistance'][:4])}")
    lines.append(f"  Support: {', '.join(fmt_price(s, d) for s in levels['support'][:4])}")

    return "\n".join(lines)


def build_swing_summary(tf_data: Dict[str, pd.DataFrame], cfg: Dict[str, Any]) -> str:
    d = cfg["decimals"]
    sig = swing_score(tf_data)

    daily_df = tf_data.get("Daily")
    h4_df = tf_data.get("4H")

    daily_close = fmt_price(float(daily_df["Close"].iloc[-1]), d) if daily_df is not None and not daily_df.empty else "N/A"
    h4_close = fmt_price(float(h4_df["Close"].iloc[-1]), d) if h4_df is not None and not h4_df.empty else "N/A"

    lines = [
        f"SWING[{cfg['display_name']}] Daily:{daily_close} 4H:{h4_close} "
        f"→ {sig['bias']} ({sig['score']:+.1f}) "
    ]
    return "\n".join(lines)


# ─── Entry point ──────────────────────────────────────────────────────

def run_swing_analysis(instrument: str = "xau", no_cache: bool = False) -> str:
    if instrument not in INSTRUMENTS:
        return f"Unknown: {instrument}. Choose: {', '.join(INSTRUMENTS)}"

    cfg = INSTRUMENTS[instrument]
    tf_data = fetch_all_timeframes(cfg, use_cache=not no_cache)
    tf_data = adjust_to_spot(tf_data, cfg)

    if not tf_data:
        return "No data. Check connection."

    for tf in ["Daily", "4H", "1H"]:
        if tf not in tf_data or tf_data[tf].empty:
            return f"ERROR: Missing {tf} data for swing."

    return build_swing_report(tf_data, cfg)


def run_swing_signal(instrument: str = "xau", no_cache: bool = False) -> str:
    if instrument not in INSTRUMENTS:
        return f"Unknown: {instrument}"

    cfg = INSTRUMENTS[instrument]
    tf_data = fetch_all_timeframes(cfg, use_cache=not no_cache)
    tf_data = adjust_to_spot(tf_data, cfg)

    if not tf_data:
        return "No data."

    return build_swing_summary(tf_data, cfg)
