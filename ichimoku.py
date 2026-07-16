#!/usr/bin/env python3
"""
Ichimoku Kinko Hyo Analysis Module
Focus: Daily, 4H (bias), 1H (entry), 15m (confirmation)
Components: Tenkan-sen, Kijun-sen, Senkou Span A/B (Kumo), Chikou Span
Strategy: Cloud breakout/retest, TK cross, Kumo twist, Chikou confirmation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

from core import (
    fetch_all_timeframes,
    calculate_indicators,
    determine_trend,
    fmt_price,
    adjust_to_spot,
    TF_ORDER,
)
from instruments import INSTRUMENTS

# ─── Ichimoku config ──────────────────────────────────────────────────

BIAS_TIMEFRAMES = ["Daily", "4H"]
ENTRY_TIMEFRAME = "1H"
CONFIRM_TIMEFRAME = "15m"

ICHIMOKU_PARAMS = {
    "tenkan": 9,
    "kijun": 26,
    "senkou_b": 52,
    "chikou_shift": 26,
}


# ─── Ichimoku calculation ────────────────────────────────────────────

def calculate_ichimoku(df: pd.DataFrame, params: Optional[dict] = None) -> pd.DataFrame:
    """Calculate all Ichimoku components on a DataFrame.

    Returns copy of df with columns:
        Tenkan, Kijun, Senkou_A, Senkou_B, Chikou
        Cloud_Top, Cloud_Bottom, Cloud_Color (Bull/Bear)
        TK_Cross (signal), Kumo_Twist (signal)
    """
    if df.empty or len(df) < 10:
        return df

    p = params or ICHIMOKU_PARAMS
    df = df.copy()
    t = p["tenkan"]
    k = p["kijun"]
    sb = p["senkou_b"]
    shift = p["chikou_shift"]

    # Tenkan-sen: (9H + 9L) / 2
    df["Tenkan"] = (df["High"].rolling(t).max() + df["Low"].rolling(t).min()) / 2

    # Kijun-sen: (26H + 26L) / 2
    df["Kijun"] = (df["High"].rolling(k).max() + df["Low"].rolling(k).min()) / 2

    # Senkou Span A: (Tenkan + Kijun) / 2, shifted forward 26 periods
    df["Senkou_A"] = ((df["Tenkan"] + df["Kijun"]) / 2).shift(shift)

    # Senkou Span B: (52H + 52L) / 2, shifted forward 26 periods
    df["Senkou_B"] = ((df["High"].rolling(sb).max() + df["Low"].rolling(sb).min()) / 2).shift(shift)

    # Chikou Span: Close shifted backward 26 periods
    df["Chikou"] = df["Close"].shift(-shift)

    # Cloud
    df["Cloud_Top"] = df[["Senkou_A", "Senkou_B"]].max(axis=1)
    df["Cloud_Bottom"] = df[["Senkou_A", "Senkou_B"]].min(axis=1)
    df["Cloud_Color"] = np.where(df["Senkou_A"] >= df["Senkou_B"], "Bull", "Bear")

    # TK Cross signal
    df["TK_Cross"] = np.where(
        (df["Tenkan"].notna()) & (df["Kijun"].notna()),
        np.where(
            (df["Tenkan"].shift(1) <= df["Kijun"].shift(1)) & (df["Tenkan"] > df["Kijun"]),
            "TK_BULL",  # Tenkan crosses above Kijun
            np.where(
                (df["Tenkan"].shift(1) >= df["Kijun"].shift(1)) & (df["Tenkan"] < df["Kijun"]),
                "TK_BEAR",  # Tenkan crosses below Kijun
                "TK_NEUTRAL"
            )
        ),
        "TK_NONE"
    )

    # Kumo Twist (Senkou A crosses Senkou B)
    df["Kumo_Twist"] = np.where(
        (df["Senkou_A"].notna()) & (df["Senkou_B"].notna()),
        np.where(
            (df["Senkou_A"].shift(1) <= df["Senkou_B"].shift(1)) & (df["Senkou_A"] > df["Senkou_B"]),
            "TWIST_BULL",
            np.where(
                (df["Senkou_A"].shift(1) >= df["Senkou_B"].shift(1)) & (df["Senkou_A"] < df["Senkou_B"]),
                "TWIST_BEAR",
                "TWIST_NONE"
            )
        ),
        "TWIST_NONE"
    )

    return df


# ─── Ichimoku signal scoring ──────────────────────────────────────────

def ichimoku_signal_df(df: pd.DataFrame, params: Optional[dict] = None) -> Dict[str, Any]:
    """Generate Ichimoku signal from a single timeframe's calculated DataFrame."""
    if df.empty or len(df) < 10:
        return {"signal": "WAIT", "score": 0, "details": []}

    df_ichi = calculate_ichimoku(df, params)
    last = df_ichi.iloc[-1]
    close = last["Close"]
    details: List[str] = []
    score = 0

    # 1. Price vs Kumo (cloud)
    cloud_top = last.get("Cloud_Top", np.nan)
    cloud_bot = last.get("Cloud_Bottom", np.nan)
    if not pd.isna(cloud_top) and not pd.isna(cloud_bot):
        if close > cloud_top:
            score += 2
            details.append(f"Price above Kumo (Bullish: {fmt_price_f(close)} > {fmt_price_f(cloud_top)})")
        elif close < cloud_bot:
            score -= 2
            details.append(f"Price below Kumo (Bearish: {fmt_price_f(close)} < {fmt_price_f(cloud_bot)})")
        else:
            details.append(f"Price inside Kumo ({fmt_price_f(cloud_bot)} - {fmt_price_f(cloud_top)})")
            # Inside cloud = neutral, but bias depends on cloud color
            if last.get("Cloud_Color") == "Bull":
                score += 1
                details.append("  Inside Bullish Kumo → slight bullish bias")
            else:
                score -= 1
                details.append("  Inside Bearish Kumo → slight bearish bias")

    # 2. Price vs Kijun-sen
    kijun = last.get("Kijun", np.nan)
    if not pd.isna(kijun):
        if close > kijun:
            score += 1
            details.append(f"Price above Kijun-sen (Bullish: {fmt_price_f(close)} > {fmt_price_f(kijun)})")
        else:
            score -= 1
            details.append(f"Price below Kijun-sen (Bearish: {fmt_price_f(close)} < {fmt_price_f(kijun)})")

    # 3. Price vs Tenkan-sen
    tenkan = last.get("Tenkan", np.nan)
    if not pd.isna(tenkan):
        if close > tenkan:
            score += 1
            details.append(f"Price above Tenkan-sen (Short-term Bullish: {fmt_price_f(close)} > {fmt_price_f(tenkan)})")
        else:
            score -= 1
            details.append(f"Price below Tenkan-sen (Short-term Bearish: {fmt_price_f(close)} < {fmt_price_f(tenkan)})")

    # 4. TK Cross (current or recent)
    tk = last.get("TK_Cross", "TK_NONE")
    if tk == "TK_BULL":
        score += 2
        details.append("TK Cross: Tenkan crossed above Kijun ✅")
    elif tk == "TK_BEAR":
        score -= 2
        details.append("TK Cross: Tenkan crossed below Kijun ❌")

    # 5. Chikou Span vs price 26 periods ago
    chikou = last.get("Chikou", np.nan)
    if not pd.isna(chikou) and len(df_ichi) >= 26:
        close_26ago = df_ichi["Close"].iloc[-26] if len(df_ichi) >= 26 else np.nan
        if not pd.isna(close_26ago):
            if chikou > close_26ago:
                score += 1
                details.append(f"Chikou above price 26 periods ago (Bullish confirmation)")
            else:
                score -= 1
                details.append(f"Chikou below price 26 periods ago (Bearish confirmation)")

    # 6. Kumo Twist
    twist = last.get("Kumo_Twist", "TWIST_NONE")
    if twist == "TWIST_BULL":
        score += 2
        details.append("Kumo Twist: Senkou A crossed above Senkou B → Bullish cloud flip 🔄")
    elif twist == "TWIST_BEAR":
        score -= 2
        details.append("Kumo Twist: Senkou A crossed below Senkou B → Bearish cloud flip 🔄")

    # Determine direction
    if score >= 4:
        signal = "STRONG BUY"
    elif score >= 2:
        signal = "BUY BIAS"
    elif score <= -4:
        signal = "STRONG SELL"
    elif score <= -2:
        signal = "SELL BIAS"
    else:
        signal = "NEUTRAL"

    return {"signal": signal, "score": score, "details": details}


def ichimoku_score(tf_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Multi-timeframe Ichimoku scoring.

    Weights: Daily x4, 4H x3, 1H x2, 15m x1.
    """
    weights = {"Daily": 4, "4H": 3, "1H": 2, "15m": 1}
    total_score = 0.0
    max_possible = 0.0
    all_details: List[str] = []
    tf_signals: Dict[str, Dict[str, Any]] = {}

    for tf_name, weight in weights.items():
        df = tf_data.get(tf_name)
        if df is None or df.empty or len(df) < 10:
            continue

        sig = ichimoku_signal_df(df)
        tf_signals[tf_name] = sig
        weighted = sig["score"] * weight
        total_score += weighted
        max_possible += 5 * weight  # max abs score per TF = 5 (approx)

        # Reduce details for summary
        if sig["details"]:
            all_details.append(f"[{tf_name}] {sig['signal']} ({sig['score']:+d}×{weight}={weighted:+d}): {sig['details'][0]}")

    # Normalize to -10..+10
    if max_possible > 0:
        normalized = (total_score / max_possible) * 10
    else:
        normalized = 0.0

    # Multi-TF alignment check
    signals_list = [s["signal"] for s in tf_signals.values()]
    buy_signals = [s for s in signals_list if "BUY" in s]
    sell_signals = [s for s in signals_list if "SELL" in s]
    neutral_count = sum(1 for s in signals_list if s == "NEUTRAL")
    if len(buy_signals) >= 3 and len(sell_signals) == 0:
        all_details.append("✅ Majority timeframes BULLISH")
    elif len(sell_signals) >= 3 and len(buy_signals) == 0:
        all_details.append("❌ Majority timeframes BEARISH")
    elif len(buy_signals) > 0 and len(sell_signals) > 0:
        all_details.append("⚠️ Conflicting BUY/SELL signals across timeframes")
    elif neutral_count >= 3:
        all_details.append("➖ Most timeframes NEUTRAL")
    else:
        all_details.append("⚠️ Mixed signals across timeframes")

    # Final bias
    if normalized >= 4:
        bias = "STRONG BUY"
    elif normalized >= 1.5:
        bias = "BUY BIAS"
    elif normalized <= -4:
        bias = "STRONG SELL"
    elif normalized <= -1.5:
        bias = "SELL BIAS"
    else:
        bias = "NEUTRAL"

    return {
        "bias": bias,
        "score": normalized,
        "details": all_details,
        "tf_signals": tf_signals,
    }


# ─── Helper ───────────────────────────────────────────────────────────

def fmt_price_f(val, decimals: int = 2) -> str:
    if pd.isna(val):
        return "N/A"
    return f"${val:.{decimals}f}"


# ─── Entry zones ──────────────────────────────────────────────────────

def ichimoku_entry_zones(tf_data: Dict[str, pd.DataFrame], bias: str, d: int) -> Optional[Dict[str, str]]:
    """Generate specific entry/exit levels based on Ichimoku."""
    h4 = tf_data.get("4H")
    h1 = tf_data.get("1H")
    if h4 is None or h1 is None or h4.empty or h1.empty:
        return None

    h4_ichi = calculate_ichimoku(h4)
    h1_ichi = calculate_ichimoku(h1)
    h4_last = h4_ichi.iloc[-1]
    h1_last = h1_ichi.iloc[-1]
    h4_close = float(h4_last["Close"])
    h1_close = float(h1_last["Close"])

    atr_h4 = float(h4_last.get("ATR", h4_last.get("ATR_14", 0)))
    if pd.isna(atr_h4) or atr_h4 == 0:
        atr_h4 = float(h4["High"].iloc[-20:].max() - h4["Low"].iloc[-20:].min()) / 20

    if "BUY" in bias:
        kijun = float(h4_last.get("Kijun", np.nan))
        tenkan_h1 = float(h1_last.get("Tenkan", np.nan))
        h4_kumo_top = float(h4_last.get("Cloud_Top", np.nan))
        h1_kumo_top = float(h1_last.get("Cloud_Top", np.nan))

        entry_zone = f"${h4_close - atr_h4 * 0.3:.{d}f} - ${h4_close:.{d}f}" if not pd.isna(h4_close) else "N/A"
        sl = f"${h4_close - atr_h4 * 1.5:.{d}f}"
        tp1 = f"${h4_close + atr_h4 * 1.5:.{d}f}"
        tp2 = f"${h4_close + atr_h4 * 3:.{d}f}"
        rr1 = f"{atr_h4 * 1.5 / atr_h4:.1f}"
        rr2 = f"{atr_h4 * 3 / atr_h4:.1f}"
        # Nearest support = Kijun or current cloud top below price
        key_support = fmt_price_f(kijun, d) if not pd.isna(kijun) else fmt_price_f(h4_kumo_top, d)
        # Nearest resistance = look up: 1H cloud top or recent 4H high
        h1_high = float(h1["High"].iloc[-5:].max())
        key_resistance = fmt_price_f(max(h1_kumo_top if not pd.isna(h1_kumo_top) else 0, h1_high), d)

        return {
            "entry_zone": entry_zone,
            "stop_loss": sl,
            "tp1": tp1 + f" (R:R = 1:{rr1})",
            "tp2": tp2 + f" (R:R = 1:{rr2})",
            "atr_4h": fmt_price_f(atr_h4, d),
            "key_support": key_support,
            "key_resistance": key_resistance,
        }

    elif "SELL" in bias:
        kijun = float(h4_last.get("Kijun", np.nan))
        kumo_bot = float(h4_last.get("Cloud_Bottom", np.nan))

        entry_zone = f"${kijun:.{d}f} - ${kijun - atr_h4 * 0.5:.{d}f}" if not pd.isna(kijun) else "N/A"
        sl = f"${kijun + atr_h4:.{d}f}" if not pd.isna(kijun) else "N/A"
        tp1 = f"${h4_close - atr_h4 * 1.5:.{d}f}"
        tp2 = f"${h4_close - atr_h4 * 3:.{d}f}"
        rr1 = f"{atr_h4 * 1.5 / atr_h4:.1f}"
        rr2 = f"{atr_h4 * 3 / atr_h4:.1f}"
        key_support = fmt_price_f(kumo_bot, d) if not pd.isna(kumo_bot) else "N/A"
        key_resistance = fmt_price_f(kijun, d) if not pd.isna(kijun) else "N/A"

        return {
            "entry_zone": entry_zone,
            "stop_loss": sl,
            "tp1": tp1 + f" (R:R = 1:{rr1})",
            "tp2": tp2 + f" (R:R = 1:{rr2})",
            "atr_4h": fmt_price_f(atr_h4, d),
            "key_support": key_support,
            "key_resistance": key_resistance,
        }

    return None


# ─── Report builder ──────────────────────────────────────────────────

def build_ichimoku_report(tf_data: Dict[str, pd.DataFrame], cfg: Dict[str, Any]) -> str:
    """Full Ichimoku analysis report."""
    d = cfg["decimals"]
    name = cfg["display_name"]

    # Validate TFs
    for tf in ["Daily", "4H", "1H"]:
        if tf not in tf_data or tf_data[tf].empty:
            return f"ERROR: Need {tf} data for Ichimoku analysis."
        if len(tf_data[tf]) < 53:
            return f"ERROR: Need at least 53 candles on {tf} for Ichimoku (got {len(tf_data[tf])})."

    lines: List[str] = []

    # ── Header ──
    lines.append("=" * 66)
    lines.append(f"  🌊 ICHIMOKU KINKO HYO — {name}")
    lines.append(f"  Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"  Tenkan:{ICHIMOKU_PARAMS['tenkan']} Kijun:{ICHIMOKU_PARAMS['kijun']} "
                 f"Senkou_B:{ICHIMOKU_PARAMS['senkou_b']} Shift:{ICHIMOKU_PARAMS['chikou_shift']}")
    lines.append("=" * 66)
    lines.append("")

    # ── Score ──
    sig = ichimoku_score(tf_data)
    bias = sig["bias"]
    bias_icon = {"STRONG BUY": "🟢🟢", "BUY BIAS": "🟢", "STRONG SELL": "🔴🔴", "SELL BIAS": "🔴", "NEUTRAL": "🟡"}.get(bias, "🟡")

    lines.append(f"【ICHIMOKU SIGNAL: {bias_icon} {bias} (score: {sig['score']:+.1f}/10)】")
    lines.append("")
    for detail in sig["details"]:
        lines.append(f"  • {detail}")
    lines.append("")

    # ── Per-timeframe Ichimoku table ──
    lines.append("【ICHIMOKU TABLE PER TIMEFRAME】")
    header = f"{'TF':<7} {'Trend':<9} {'Price':<13} {'Tenkan':<13} {'Kijun':<13} {'Senkou_A':<13} {'Senkou_B':<13} {'Cloud':<6} {'RSI':<6}"
    lines.append(header)
    lines.append("-" * len(header))

    for tf_name in ["Daily", "4H", "1H", "15m"]:
        df = tf_data.get(tf_name)
        if df is None or df.empty or len(df) < 10:
            continue

        df_ichi = calculate_ichimoku(df)
        last = df_ichi.iloc[-1]
        trend, _ = determine_trend(df)
        close = last["Close"]
        tenkan = last.get("Tenkan", np.nan)
        kijun = last.get("Kijun", np.nan)
        senkou_a = last.get("Senkou_A", np.nan)
        senkou_b = last.get("Senkou_B", np.nan)
        cloud_color = last.get("Cloud_Color", "N/A")
        rsi = last.get("RSI_14", np.nan)

        # Price vs cloud icon
        cloud_top = last.get("Cloud_Top", np.nan)
        cloud_bot = last.get("Cloud_Bottom", np.nan)
        if not pd.isna(cloud_top) and not pd.isna(cloud_bot):
            if close > cloud_top:
                cloud_icon = "☀️Above"
            elif close < cloud_bot:
                cloud_icon = "🌧️Below"
            else:
                cloud_icon = "☁️Inside"
        else:
            cloud_icon = "N/A"

        rsi_s = f"{rsi:.1f}" if not pd.isna(rsi) else "N/A"

        lines.append(
            f"{tf_name:<7} {trend:<9} {fmt_price_f(close, d):<13} "
            f"{fmt_price_f(tenkan, d):<13} {fmt_price_f(kijun, d):<13} "
            f"{fmt_price_f(senkou_a, d):<13} {fmt_price_f(senkou_b, d):<13} "
            f"{cloud_icon:<6} {rsi_s:<6}"
        )

    lines.append("")

    # ── Latest TK Cross & Kumo Twist ──
    lines.append("【TK CROSS & KUMO TWIST】")
    for tf_name in ["Daily", "4H", "1H", "15m"]:
        df = tf_data.get(tf_name)
        if df is None or df.empty or len(df) < 10:
            continue
        df_ichi = calculate_ichimoku(df)
        last = df_ichi.iloc[-1]

        tk = last.get("TK_Cross", "N/A")
        twist = last.get("Kumo_Twist", "N/A")
        tenkan = last.get("Tenkan", np.nan)
        kijun = last.get("Kijun", np.nan)

        tk_icon = {"TK_BULL": "🟢 Bullish cross", "TK_BEAR": "🔴 Bearish cross", "TK_NEUTRAL": "⚪ No cross"}.get(tk, f"({tk})")
        twist_icon = {"TWIST_BULL": "🟢 Bullish twist", "TWIST_BEAR": "🔴 Bearish twist", "TWIST_NONE": "⚪ No twist"}.get(twist, f"({twist})")

        lines.append(f"  [{tf_name}] TK={tk_icon} | "
                     f"Tenkan={fmt_price_f(tenkan, d)} Kijun={fmt_price_f(kijun, d)} | "
                     f"Kumo={twist_icon}")

    lines.append("")

    # ── Cloud projection (next ~26 periods) ──
    lines.append("【KUMO PROJECTION (Forward)】")
    for tf_name in ["Daily", "4H", "1H"]:
        df = tf_data.get(tf_name)
        if df is None or df.empty or len(df) < 53:
            continue
        df_ichi = calculate_ichimoku(df)

        # Check if cloud is thickening or thinning
        # The current cloud edge is determined by Senkou_A and Senkou_B values
        # For the "future" cloud, we look at the most recent values that will be shifted forward
        last_senkou_a = df_ichi["Senkou_A"].iloc[-1] if not df_ichi["Senkou_A"].iloc[-26:].isna().all() else np.nan
        last_senkou_b = df_ichi["Senkou_B"].iloc[-1] if not df_ichi["Senkou_B"].iloc[-26:].isna().all() else np.nan

        # Check cloud slope: look at Senkou_A trend over last few periods
        sa_vals = df_ichi["Senkou_A"].dropna()
        if len(sa_vals) >= 5:
            sa_slope = sa_vals.iloc[-1] - sa_vals.iloc[-5]
            slope_dir = "Rising ↗️" if sa_slope > 0 else "Falling ↘️" if sa_slope < 0 else "Flat →"
        else:
            slope_dir = "N/A"

        # Cloud thickness
        if not pd.isna(last_senkou_a) and not pd.isna(last_senkou_b):
            thickness = abs(last_senkou_a - last_senkou_b)
            thick_desc = f"Thick ({fmt_price_f(thickness, d)})" if thickness > 0 else "Thin"
        else:
            thick_desc = "N/A"

        lines.append(f"  [{tf_name}] Cloud slope: {slope_dir} | Thickness: {thick_desc}")
    lines.append("")

    # ── Entry zones ──
    zones = ichimoku_entry_zones(tf_data, bias, d) if bias != "NEUTRAL" else None

    if zones:
        lines.append("【ENTRY & EXIT PLAN (Ichimoku)】")
        lines.append(f"  Entry Zone:    {zones['entry_zone']}")
        lines.append(f"  Stop Loss:     {zones['stop_loss']}")
        lines.append(f"  Take Profit 1: {zones['tp1']}")
        lines.append(f"  Take Profit 2: {zones['tp2']}")
        lines.append(f"  ATR(4H):       {zones['atr_4h']}")
        lines.append(f"  Key Support:   {zones['key_support']}")
        lines.append(f"  Key Resistance:{zones['key_resistance']}")
        lines.append("")
        lines.append("  📋 Ichimoku Trading Rules:")
        lines.append("  • BUY: Price above Kumo + Tenkan > Kijun (TK Bull) + Chikou confirming")
        lines.append("  • SELL: Price below Kumo + Tenkan < Kijun (TK Bear) + Chikou confirming")
        lines.append("  • Kumo filter: Only trade direction of cloud color (bullish cloud = long only)")
        lines.append("  • Kijun-sen as trailing stop in trend direction")
        lines.append("  • Avoid trading inside Kumo (wait for breakout/breakdown)")
        lines.append("  • Kumo Twist = potential trend reversal signal")
    else:
        lines.append("【NO CLEAR ENTRY】")
        lines.append("  🟡 Ichimoku signals mixed or neutral. Wait for clearer setup.")
        lines.append("  • Chờ giá breakout khỏi Kumo")
        lines.append("  • Chờ TK Cross rõ ràng theo hướng cloud")
        lines.append("  • Chờ Chikou Span xác nhận")

    lines.append("")
    lines.append("=" * 66)

    return "\n".join(lines)


# ─── Quick signal ─────────────────────────────────────────────────────

def build_ichimoku_summary(tf_data: Dict[str, pd.DataFrame], cfg: Dict[str, Any]) -> str:
    """Compact one-line Ichimoku signal."""
    d = cfg["decimals"]
    sig = ichimoku_score(tf_data)

    daily = tf_data.get("Daily")
    h4 = tf_data.get("4H")

    daily_close = fmt_price_f(float(daily["Close"].iloc[-1]), d) if daily is not None and not daily.empty else "N/A"
    h4_close = fmt_price_f(float(h4["Close"].iloc[-1]), d) if h4 is not None and not h4.empty else "N/A"

    # Cloud status
    cloud_status = ""
    if h4 is not None and not h4.empty:
        h4_ichi = calculate_ichimoku(h4)
        h4_last = h4_ichi.iloc[-1]
        ct = h4_last.get("Cloud_Top", np.nan)
        cb = h4_last.get("Cloud_Bottom", np.nan)
        close_h4 = float(h4_last["Close"])
        if not pd.isna(ct) and not pd.isna(cb):
            if close_h4 > ct:
                cloud_status = "☀️AboveCloud"
            elif close_h4 < cb:
                cloud_status = "🌧️BelowCloud"
            else:
                cloud_status = "☁️InCloud"

    lines = [
        f"ICHIMOKU[{cfg['display_name']}] Daily:{daily_close} 4H:{h4_close} {cloud_status} "
        f"→ {sig['bias']} ({sig['score']:+.1f}) "
    ]
    return "\n".join(lines)


# ─── Entry points ────────────────────────────────────────────────────

def run_ichimoku_analysis(instrument: str = "xau", no_cache: bool = False) -> str:
    if instrument not in INSTRUMENTS:
        return f"Unknown: {instrument}. Choose: {', '.join(INSTRUMENTS)}"

    cfg = INSTRUMENTS[instrument]
    tf_data = fetch_all_timeframes(cfg, use_cache=not no_cache)

    if not tf_data:
        return "No data. Check connection."

    tf_data = adjust_to_spot(tf_data, cfg)

    for tf in ["Daily", "4H", "1H"]:
        if tf not in tf_data or tf_data[tf].empty:
            return f"ERROR: Missing {tf} data for Ichimoku."
        if len(tf_data[tf]) < 53:
            return f"ERROR: Need >=53 candles on {tf} (got {len(tf_data[tf])}). Try --no-cache."

    return build_ichimoku_report(tf_data, cfg)


def run_ichimoku_signal(instrument: str = "xau", no_cache: bool = False) -> str:
    if instrument not in INSTRUMENTS:
        return f"Unknown: {instrument}"

    cfg = INSTRUMENTS[instrument]
    tf_data = fetch_all_timeframes(cfg, use_cache=not no_cache)

    if not tf_data:
        return "No data."

    tf_data = adjust_to_spot(tf_data, cfg)

    return build_ichimoku_summary(tf_data, cfg)
