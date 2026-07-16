#!/usr/bin/env python3
"""
Position Trading Analysis Module
Focus: Weekly, Daily (bias), 4H (entry)
Hold: Vài tuần đến vài tháng
Strategy: Macro trend-following, 200 SMA, long-term S/R, COT-style sentiment proxy
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta

from core import (
    fetch_all_timeframes,
    fetch_spot_price,
    calculate_indicators,
    determine_trend,
    fmt_price,
    TF_ORDER,
)
from instruments import INSTRUMENTS

# ─── Position config ───────────────────────────────────────────────────

POSITION_BIAS_TF = "Daily"     # Bias chính
POSITION_STRUCTURE_TF = "4H"   # Cấu trúc trung hạn
POSITION_ENTRY_TF = "4H"       # TF entry (có thể là Daily nếu đủ kiên nhẫn)

POSITION_ATR_SL = 3.0          # SL = 3x ATR(Daily)
POSITION_ATR_TP1 = 4.0         # TP1 = 4x ATR
POSITION_ATR_TP2 = 8.0         # TP2 = 8x ATR
POSITION_ATR_TP3 = 12.0        # TP3 = 12x ATR (runner)

POSITION_RISK_PCT = 1.0         # Risk per trade: 1% of account
POSITION_MIN_RR = 2.0           # Minimum R:R to take trade


# ─── Long-term trend strength ──────────────────────────────────────────

def macro_trend_score(df_daily: pd.DataFrame) -> Dict[str, Any]:
    """Đánh giá sức mạnh xu hướng dài hạn từ Daily."""
    if df_daily.empty or len(df_daily) < 50:
        return {"trend": "WAIT", "score": 0, "details": ["Insufficient data"]}

    last = df_daily.iloc[-1]
    close = float(last["Close"])
    score = 0.0
    details: List[str] = []

    # 1. SMA 50 vs SMA 200 (Golden/Death Cross)
    sma50 = float(last.get("SMA_50", np.nan)) if not pd.isna(last.get("SMA_50", np.nan)) else None
    sma200 = float(last.get("SMA_200", np.nan)) if not pd.isna(last.get("SMA_200", np.nan)) else None

    if sma50 is not None and sma200 is not None:
        if sma50 > sma200:
            score += 2.5
            details.append(f"SMA50({fmt_price(sma50, 2)}) > SMA200({fmt_price(sma200, 2)}) — GOLDEN CROSS [+2.5]")
            # Check if recently crossed (last 20 bars)
            sma50_20ago = float(df_daily["SMA_50"].iloc[-20]) if len(df_daily) >= 20 and not pd.isna(df_daily["SMA_50"].iloc[-20]) else None
            sma200_20ago = float(df_daily["SMA_200"].iloc[-20]) if len(df_daily) >= 20 and not pd.isna(df_daily["SMA_200"].iloc[-20]) else None
            if sma50_20ago is not None and sma200_20ago is not None and sma50_20ago < sma200_20ago:
                score += 1.5
                details.append("  → Recent Golden Cross (last 20 bars) [+1.5]")
        else:
            score -= 2.5
            details.append(f"SMA50 < SMA200 — DEATH CROSS [-2.5]")
            sma50_20ago = float(df_daily["SMA_50"].iloc[-20]) if len(df_daily) >= 20 and not pd.isna(df_daily["SMA_50"].iloc[-20]) else None
            sma200_20ago = float(df_daily["SMA_200"].iloc[-20]) if len(df_daily) >= 20 and not pd.isna(df_daily["SMA_200"].iloc[-20]) else None
            if sma50_20ago is not None and sma200_20ago is not None and sma50_20ago > sma200_20ago:
                score -= 1.5
                details.append("  → Recent Death Cross (last 20 bars) [-1.5]")

    # 2. Price vs 200 SMA (core position filter)
    if sma200 is not None:
        pct_from_200 = (close - sma200) / sma200 * 100
        if pct_from_200 > 10:
            score += 1
            details.append(f"Price {pct_from_200:+.1f}% vs SMA200 — strong uptrend [+1]")
        elif pct_from_200 > 0:
            score += 0.5
            details.append(f"Price {pct_from_200:+.1f}% vs SMA200 — above 200 [+0.5]")
        elif pct_from_200 > -10:
            score -= 0.5
            details.append(f"Price {pct_from_200:+.1f}% vs SMA200 — below 200 [-0.5]")
        else:
            score -= 1
            details.append(f"Price {pct_from_200:+.1f}% vs SMA200 — strong downtrend [-1]")

    # 3. ADX-style trend strength (using ATR/price ratio + directional movement)
    atr = float(last.get("ATR", np.nan)) if not pd.isna(last.get("ATR", np.nan)) else None
    if atr is not None and close > 0:
        atr_pct = (atr / close) * 100

        # Check if ATR is expanding (trending) or contracting (ranging)
        atr_20ago = float(df_daily["ATR"].iloc[-20]) if len(df_daily) >= 20 and not pd.isna(df_daily["ATR"].iloc[-20]) else atr
        atr_change = (atr - atr_20ago) / atr_20ago * 100 if atr_20ago > 0 else 0

        if atr_change > 15:
            details.append(f"ATR expanding +{atr_change:.0f}% — volatility increasing [trending]")
        elif atr_change < -15:
            details.append(f"ATR contracting {atr_change:.0f}% — volatility decreasing [consolidating]")

    # 4. Higher Highs / Higher Lows structure (last 3 months ≈ 60 bars)
    lookback = min(60, len(df_daily))
    highs = df_daily["High"].iloc[-lookback:]
    lows = df_daily["Low"].iloc[-lookback:]

    # Split into 3 segments
    seg = lookback // 3
    h1, h2, h3 = highs.iloc[:seg].max(), highs.iloc[seg:2*seg].max(), highs.iloc[2*seg:].max()
    l1, l2, l3 = lows.iloc[:seg].min(), lows.iloc[seg:2*seg].min(), lows.iloc[2*seg:].min()

    if h1 < h2 < h3 and l1 < l2 < l3:
        score += 2
        details.append("HH/HL structure — bullish trend intact [+2]")
    elif h1 > h2 > h3 and l1 > l2 > l3:
        score -= 2
        details.append("LH/LL structure — bearish trend intact [-2]")
    elif h1 < h2 and h2 > h3 and l1 < l2 and l2 < l3:
        score += 0.5
        details.append("Potential HH but recent pullback — bullish but cautious [+0.5]")
    elif h1 > h2 and h2 < h3 and l1 > l2 and l2 > l3:
        score -= 0.5
        details.append("Potential LL but recent bounce — bearish but cautious [-0.5]")

    # 5. RSI trend (long-term RSI zone)
    rsi = float(last.get("RSI_14", np.nan)) if not pd.isna(last.get("RSI_14", np.nan)) else None
    if rsi is not None:
        if 40 <= rsi <= 60:
            details.append(f"RSI(14)={rsi:.1f} — neutral zone")
        elif rsi > 60:
            score += 0.5
            details.append(f"RSI(14)={rsi:.1f} — bullish momentum [+0.5]")
            if rsi > 70:
                details.append(f"  ⚠️ RSI > 70 — overbought, wait for pullback")
        else:
            score -= 0.5
            details.append(f"RSI(14)={rsi:.1f} — bearish momentum [-0.5]")
            if rsi < 30:
                details.append(f"  ⚠️ RSI < 30 — oversold, wait for bounce")

    # 6. MACD on Daily
    macd = float(last.get("MACD", np.nan)) if not pd.isna(last.get("MACD", np.nan)) else None
    macd_sig = float(last.get("MACD_Signal", np.nan)) if not pd.isna(last.get("MACD_Signal", np.nan)) else None
    if macd is not None and macd_sig is not None:
        if macd > macd_sig and macd > 0:
            score += 1
            details.append(f"MACD bullish above zero [+1]")
        elif macd > macd_sig:
            score += 0.5
            details.append(f"MACD bullish below zero [+0.5]")
        elif macd < macd_sig and macd < 0:
            score -= 1
            details.append(f"MACD bearish below zero [-1]")
        else:
            score -= 0.5
            details.append(f"MACD bearish above zero [-0.5]")

    # Determine final bias
    if score >= 5:
        bias = "STRONG BUY"
    elif score >= 2:
        bias = "BUY BIAS"
    elif score <= -5:
        bias = "STRONG SELL"
    elif score <= -2:
        bias = "SELL BIAS"
    else:
        bias = "WAIT"

    return {
        "trend": "UP" if score > 0 else "DOWN" if score < 0 else "SIDEWAYS",
        "score": round(score, 1),
        "bias": bias,
        "sma200": sma200,
        "pct_from_200": round(pct_from_200, 1) if sma200 else None,
        "details": details,
    }


# ─── Long-term S/R zones ───────────────────────────────────────────────

def longterm_sr_zones(df_daily: pd.DataFrame, decimals: int) -> Dict[str, Any]:
    """Xác định vùng hỗ trợ/kháng cự dài hạn."""
    if df_daily.empty or len(df_daily) < 50:
        return {"resistance": [], "support": [], "monthly_range": None}

    # Monthly pivots (dùng 20 ngày ≈ 1 tháng)
    monthly = df_daily.iloc[-20:]
    m_high = float(monthly["High"].max())
    m_low = float(monthly["Low"].min())
    m_close = float(monthly["Close"].iloc[-1])

    # Quarterly levels (60 ngày)
    quarterly = df_daily.iloc[-60:] if len(df_daily) >= 60 else df_daily
    q_high = float(quarterly["High"].max())
    q_low = float(quarterly["Low"].min())

    # Yearly levels (250 ngày)
    yearly = df_daily.iloc[-250:] if len(df_daily) >= 250 else df_daily
    y_high = float(yearly["High"].max())
    y_low = float(yearly["Low"].min())

    # Classic pivot points from monthly
    pivot = round((m_high + m_low + m_close) / 3, decimals)
    r1 = round(2 * pivot - m_low, decimals)
    r2 = round(pivot + (m_high - m_low), decimals)
    r3 = round(m_high + 2 * (pivot - m_low), decimals)
    s1 = round(2 * pivot - m_high, decimals)
    s2 = round(pivot - (m_high - m_low), decimals)
    s3 = round(m_low - 2 * (m_high - pivot), decimals)

    price = float(df_daily["Close"].iloc[-1])

    # SMA levels
    last = df_daily.iloc[-1]
    sma_levels = {}
    for sma_name, label in [("SMA_20", "SMA20"), ("SMA_50", "SMA50"), ("SMA_200", "SMA200")]:
        val = last.get(sma_name, np.nan)
        if not pd.isna(val):
            sma_levels[label] = round(float(val), decimals)

    # Separate into S/R relative to price
    all_levels = [pivot, r1, r2, r3, s1, s2, s3, q_high, q_low, y_high, y_low] + list(sma_levels.values())
    all_levels = sorted(set(round(x, decimals) for x in all_levels))

    resistance = [x for x in all_levels if x > price * 1.002][:5]  # 0.2% buffer
    support = [x for x in all_levels if x < price * 0.998][-5:]
    support.reverse()

    return {
        "resistance": resistance,
        "support": support,
        "monthly_range": f"{fmt_price(m_low, decimals)} – {fmt_price(m_high, decimals)}",
        "quarterly_range": f"{fmt_price(q_low, decimals)} – {fmt_price(q_high, decimals)}",
        "yearly_range": f"{fmt_price(y_low, decimals)} – {fmt_price(y_high, decimals)}",
        "monthly_pivot": fmt_price(pivot, decimals),
        "monthly_r1": fmt_price(r1, decimals),
        "monthly_s1": fmt_price(s1, decimals),
        "sma_levels": sma_levels,
        "price": round(price, decimals),
    }


# ─── Multi-TF confluence for position ──────────────────────────────────

def position_confluence(tf_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Đánh giá độ hội tụ đa khung thời gian cho position trade."""
    trends: Dict[str, str] = {}
    scores: Dict[str, int] = {}
    details: List[str] = []

    weight_map = {"Daily": 4.0, "4H": 3.0, "1H": 1.0}

    total_weighted = 0.0
    total_weight = 0.0

    for tf_name in ["Daily", "4H", "1H"]:
        df = tf_data.get(tf_name)
        if df is None or df.empty:
            continue
        trend, score = determine_trend(df)
        trends[tf_name] = trend
        scores[tf_name] = score
        w = weight_map.get(tf_name, 1)
        total_weighted += score * w
        total_weight += w
        details.append(f"  {tf_name}: {trend} (score: {score:+d}, weight: x{w})")

    avg_score = round(total_weighted / total_weight, 1) if total_weight > 0 else 0

    # Confluence strength
    trend_values = list(trends.values())
    if len(trend_values) >= 2 and all(t == trend_values[0] for t in trend_values):
        if trend_values[0] == "UP":
            confluence = "STRONG ALIGNMENT — All TFs bullish"
        elif trend_values[0] == "DOWN":
            confluence = "STRONG ALIGNMENT — All TFs bearish"
        else:
            confluence = "All TFs sideways — no trade"
    elif len(trend_values) >= 2:
        up_count = sum(1 for t in trend_values if t == "UP")
        down_count = sum(1 for t in trend_values if t == "DOWN")
        if up_count > down_count:
            confluence = f"BULLISH LEAN — {'▲' * up_count}{'▼' * down_count}"
        elif down_count > up_count:
            confluence = f"BEARISH LEAN — {'▲' * up_count}{'▼' * down_count}"
        else:
            confluence = "MIXED — conflicting signals"
    else:
        confluence = "Insufficient data"

    return {
        "avg_score": avg_score,
        "confluence": confluence,
        "details": details,
        "trends": trends,
        "scores": scores,
    }


# ─── Position entry/exit zones ─────────────────────────────────────────

def position_entry_zones(
    tf_data: Dict[str, pd.DataFrame],
    bias: str,
    decimals: int,
) -> Optional[Dict[str, Any]]:
    """Tính entry/exit cho position trade dựa trên Daily ATR."""
    daily = tf_data.get("Daily")
    if daily is None or daily.empty:
        return None

    last = daily.iloc[-1]
    price = float(last["Close"])
    atr_daily = float(last.get("ATR", np.nan)) if not pd.isna(last.get("ATR", np.nan)) else float(price * 0.01)

    sr = longterm_sr_zones(daily, decimals)

    if "BUY" in bias:
        # Entry: pullback to nearest support or SMA
        supports = sr["support"]
        sma_levels = sr.get("sma_levels", {})

        key_levels = supports[:3]
        for sma_val in sma_levels.values():
            if sma_val < price and sma_val not in key_levels:
                key_levels.append(sma_val)

        key_levels = sorted(set(key_levels), reverse=True)
        entry_bottom = key_levels[0] if key_levels else price - atr_daily * 1.5
        entry_top = price

        sl = round(entry_bottom - atr_daily * POSITION_ATR_SL, decimals)
        tp1 = round(price + atr_daily * POSITION_ATR_TP1, decimals)
        tp2 = round(price + atr_daily * POSITION_ATR_TP2, decimals)
        tp3 = round(price + atr_daily * POSITION_ATR_TP3, decimals)

        # Adjust TPs to resistance
        resistances = sr["resistance"]
        for r in sorted(resistances):
            if tp1 < r < tp2:
                tp2 = r
                break

        risk = price - sl
        rr1 = round((tp1 - price) / risk, 1) if risk > 0 else 0
        rr2 = round((tp2 - price) / risk, 1) if risk > 0 else 0
        rr3 = round((tp3 - price) / risk, 1) if risk > 0 else 0
    else:
        resistances = sr["resistance"]
        sma_levels = sr.get("sma_levels", {})

        key_levels = resistances[:3]
        for sma_val in sma_levels.values():
            if sma_val > price and sma_val not in key_levels:
                key_levels.append(sma_val)

        key_levels = sorted(set(key_levels))
        entry_top = key_levels[0] if key_levels else price + atr_daily * 1.5
        entry_bottom = price

        sl = round(entry_top + atr_daily * POSITION_ATR_SL, decimals)
        tp1 = round(price - atr_daily * POSITION_ATR_TP1, decimals)
        tp2 = round(price - atr_daily * POSITION_ATR_TP2, decimals)
        tp3 = round(price - atr_daily * POSITION_ATR_TP3, decimals)

        supports = sr["support"]
        for s in sorted(supports, reverse=True):
            if tp1 > s > tp2:
                tp2 = s
                break

        risk = sl - price
        rr1 = round((price - tp1) / risk, 1) if risk > 0 else 0
        rr2 = round((price - tp2) / risk, 1) if risk > 0 else 0
        rr3 = round((price - tp3) / risk, 1) if risk > 0 else 0

    # Position sizing
    position_pct = POSITION_RISK_PCT

    return {
        "entry_zone": f"{fmt_price(entry_bottom, decimals)} – {fmt_price(entry_top, decimals)}",
        "stop_loss": fmt_price(sl, decimals),
        "tp1": fmt_price(tp1, decimals),
        "tp2": fmt_price(tp2, decimals),
        "tp3": fmt_price(tp3, decimals),
        "rr1": rr1,
        "rr2": rr2,
        "rr3": rr3,
        "atr_daily": fmt_price(atr_daily, decimals),
        "risk_per_trade": f"{position_pct}%",
        "sl_distance_pct": round(abs(price - sl) / price * 100, 2),
    }


# ─── Build position report ─────────────────────────────────────────────

def build_position_report(tf_data: Dict[str, pd.DataFrame], cfg: Dict[str, Any]) -> str:
    d = cfg["decimals"]
    name = cfg["display_name"]

    for tf in ["Daily", "4H", "1H"]:
        if tf not in tf_data or tf_data[tf].empty:
            return f"ERROR: Need {tf} data for position trading."

    # Macro trend
    macro = macro_trend_score(tf_data["Daily"])
    bias = macro["bias"]

    # Confluence
    conf = position_confluence(tf_data)

    # Long-term S/R
    sr = longterm_sr_zones(tf_data["Daily"], d)

    # Entry zones
    zones = position_entry_zones(tf_data, bias, d) if bias != "WAIT" else None

    # ── Build report ──
    lines: List[str] = []
    lines.append("=" * 68)
    lines.append(f"  🏛️  POSITION TRADING ANALYSIS — {name}")
    lines.append(f"  Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"  Style: Long-term (hold weeks to months)")
    lines.append(f"  Risk/Trade: {POSITION_RISK_PCT}%  |  Min R:R: 1:{POSITION_MIN_RR}")
    lines.append("=" * 68)
    lines.append("")

    # Signal
    bias_icon = {"STRONG BUY": "🟢", "BUY BIAS": "🟢", "STRONG SELL": "🔴", "SELL BIAS": "🔴"}.get(bias, "🟡")
    lines.append(f"【POSITION SIGNAL: {bias_icon} {bias} (score: {macro['score']:+.1f})】")
    lines.append("")

    # Macro trend breakdown
    lines.append("【MACRO TREND ANALYSIS — Daily】")
    if macro["sma200"] is not None:
        lines.append(f"  SMA200:  {fmt_price(macro['sma200'], d)}")
    if macro["pct_from_200"] is not None:
        lines.append(f"  Price vs SMA200: {macro['pct_from_200']:+.1f}%")
    lines.append(f"  Score Breakdown:")
    for detail in macro["details"]:
        lines.append(f"    {detail}")
    lines.append("")

    # Multi-TF Confluence
    lines.append("【MULTI-TIMEFRAME CONFLUENCE】")
    lines.append(f"  {conf['confluence']}")
    lines.append(f"  Weighted Score: {conf['avg_score']:+.1f}")
    for detail in conf["details"]:
        lines.append(detail)
    lines.append("")

    # Long-term S/R zones
    lines.append("【LONG-TERM SUPPORT & RESISTANCE】")
    lines.append(f"  Monthly Range:   {sr['monthly_range']}")
    lines.append(f"  Quarterly Range:  {sr['quarterly_range']}")
    lines.append(f"  Yearly Range:     {sr['yearly_range']}")
    lines.append(f"  Monthly Pivot:    {sr['monthly_pivot']}  (R1: {sr['monthly_r1']} | S1: {sr['monthly_s1']})")
    lines.append(f"  Resistance:       {', '.join(fmt_price(r, d) for r in sr['resistance'][:5]) or 'N/A'}")
    lines.append(f"  Support:          {', '.join(fmt_price(s, d) for s in sr['support'][:5]) or 'N/A'}")

    sma_str = "  ".join(f"{k}: {fmt_price(v, d)}" for k, v in sr.get("sma_levels", {}).items())
    if sma_str:
        lines.append(f"  SMA Levels:       {sma_str}")
    lines.append("")

    # Entry/Exit Plan
    if zones:
        lines.append("【POSITION TRADE PLAN】")
        lines.append(f"  Entry Zone:       {zones['entry_zone']}")
        lines.append(f"  Stop Loss:        {zones['stop_loss']}  ({zones['sl_distance_pct']}% from price)")
        lines.append(f"  Take Profit 1:    {zones['tp1']}  (R:R = 1:{zones['rr1']}) — Close 30%")
        lines.append(f"  Take Profit 2:    {zones['tp2']}  (R:R = 1:{zones['rr2']}) — Close 30%")
        lines.append(f"  Take Profit 3:    {zones['tp3']}  (R:R = 1:{zones['rr3']}) — Runner 40%")
        lines.append(f"  ATR(Daily):       {zones['atr_daily']}")
        lines.append(f"  Risk/Trade:       {zones['risk_per_trade']}")
        lines.append("")
        lines.append("  📋 Position Trading Rules:")
        lines.append("  • Entry: Scale-in 50% now, 50% if price retests entry zone")
        lines.append("  • SL: Wide stop based on structure — không chạm SL nghĩa là trend intact")
        lines.append("  • Add: Có thể thêm vị thế khi pullback về SMA50/SMA200 (pyramiding)")
        lines.append("  • Reduce: Đóng 1/3 khi chạm TP1, dời SL về hòa vốn")
        lines.append("  • Trail: Dùng SMA50 trên Daily để trail phần runner")
        lines.append("  • News: Tránh mở lệnh mới 2 ngày trước FOMC/NFP/CPI")
        lines.append("  • Journal: Ghi lại lý do entry, chụp chart Daily + Weekly")
        lines.append("  • Max positions: 2-3 lệnh cùng lúc để đa dạng hóa")
    else:
        lines.append("【NO POSITION SETUP】")
        lines.append("  ⏸️  Chưa có setup position rõ ràng. Điều kiện cần:")
        lines.append("  • Daily trend rõ ràng (không sideways)")
        lines.append("  • SMA50 và SMA200 có hướng dốc rõ rệt")
        lines.append("  • Giá ở gần vùng value (gần SMA50 hoặc SMA200)")
        lines.append("  • RSI không quá overbought/oversold")
        lines.append("")
        lines.append("  📊 Action Plan:")
        lines.append("  • Theo dõi Weekly chart để xác nhận macro trend")
        lines.append("  • Đặt alert tại SMA200 và key S/R levels")
        lines.append("  • Chờ Daily close xác nhận breakout/pullback")
        lines.append("  • Khi có setup: scale-in từ từ, không FOMO")

    lines.append("")
    lines.append("=" * 68)
    return "\n".join(lines)


# ─── Quick signal ─────────────────────────────────────────────────────

def build_position_summary(tf_data: Dict[str, pd.DataFrame], cfg: Dict[str, Any]) -> str:
    macro = macro_trend_score(tf_data.get("Daily", pd.DataFrame()))
    daily_last = tf_data.get("Daily", pd.DataFrame()).iloc[-1] if tf_data.get("Daily") is not None and not tf_data["Daily"].empty else None

    price_str = fmt_price(float(daily_last["Close"]), cfg["decimals"]) if daily_last is not None else "N/A"
    sma200_str = fmt_price(macro["sma200"], cfg["decimals"]) if macro["sma200"] else "N/A"
    pct_str = f"{macro['pct_from_200']:+.1f}%" if macro["pct_from_200"] is not None else "N/A"

    return (
        f"POSITION[{cfg['display_name']}] Price:{price_str} SMA200:{sma200_str} "
        f"({pct_str}) → {macro['bias']} ({macro['score']:+.1f})"
    )


# ─── Entry point ──────────────────────────────────────────────────────

def run_position_analysis(instrument: str = "xau", no_cache: bool = False) -> str:
    if instrument not in INSTRUMENTS:
        return f"Unknown: {instrument}"

    cfg = INSTRUMENTS[instrument]
    tf_data = fetch_all_timeframes(cfg, use_cache=not no_cache)

    if not tf_data:
        return "No data."

    for tf in ["Daily", "4H", "1H"]:
        if tf not in tf_data or tf_data[tf].empty:
            return f"ERROR: Missing {tf} data."

    return build_position_report(tf_data, cfg)


def run_position_signal(instrument: str = "xau", no_cache: bool = False) -> str:
    if instrument not in INSTRUMENTS:
        return f"Unknown: {instrument}"

    cfg = INSTRUMENTS[instrument]
    tf_data = fetch_all_timeframes(cfg, use_cache=not no_cache)

    if not tf_data:
        return "No data."

    return build_position_summary(tf_data, cfg)
