#!/usr/bin/env python3
"""
Day Trading Analysis Module
Focus: 4H (bias), 1H (structure), 15m (entry), 5m (precision)
Hold: Vài phút đến vài giờ, đóng hết cuối ngày
Strategy: Intraday momentum, volume profile, VWAP, opening range breakout
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from core import (
    fetch_all_timeframes,
    fetch_spot_price,
    calculate_indicators,
    determine_trend,
    fmt_price,
    TF_ORDER,
)
from instruments import INSTRUMENTS

# ─── Day Trade config ──────────────────────────────────────────────────

DAY_BIAS_TF = "4H"         # Bias chính trong ngày
DAY_STRUCTURE_TF = "1H"    # Cấu trúc intraday
DAY_ENTRY_TF = "15m"       # TF tìm entry
DAY_PRECISION_TF = "5m"    # TF căn entry chính xác

DAY_ATR_SL = 1.5            # SL = 1.5x ATR(15m)
DAY_ATR_TP1 = 2.0           # TP1 = 2.0x ATR
DAY_ATR_TP2 = 3.5           # TP2 = 3.5x ATR


# ─── Intraday VWAP ────────────────────────────────────────────────────

def calculate_vwap(df: pd.DataFrame) -> Optional[float]:
    """Tính VWAP (Volume Weighted Average Price) cho phiên hiện tại."""
    if df.empty or "Volume" not in df.columns:
        return None

    # Dùng 1 ngày gần nhất (approx intraday)
    today = df.iloc[-96:] if len(df) >= 96 else df  # ~24h của 15m
    typical_price = (today["High"] + today["Low"] + today["Close"]) / 3
    vol = today["Volume"]

    if vol.sum() == 0:
        return None

    vwap = float((typical_price * vol).sum() / vol.sum())
    return vwap


# ─── Opening Range ─────────────────────────────────────────────────────

def opening_range_levels(df_15m: pd.DataFrame) -> Optional[Dict[str, float]]:
    """Tính Opening Range (6 nến 15m đầu phiên = 1.5h đầu)."""
    if df_15m.empty or len(df_15m) < 6:
        return None

    or_candles = df_15m.iloc[-6:]  # Gần đúng
    or_high = float(or_candles["High"].max())
    or_low = float(or_candles["Low"].min())
    or_range = or_high - or_low
    price = float(df_15m["Close"].iloc[-1])

    return {
        "or_high": round(or_high, 2),
        "or_low": round(or_low, 2),
        "or_range": round(or_range, 2),
        "price": round(price, 2),
        "position": "above" if price > or_high else "below" if price < or_low else "inside",
    }


# ─── Volume Profile ──────────────────────────────────────────────────
# Sử dụng volume_profile.py (improved)

def volume_profile_poc(df_1h: pd.DataFrame, bins: int = 10) -> Optional[Dict[str, Any]]:
    """Tìm POC (Point of Control) từ 1H data — wrapper cho volume_profile module."""
    from volume_profile import analyze_volume_profile
    try:
        vp = analyze_volume_profile(df_1h, tf_name="1H", bins=bins, use_dynamic_bins=False)
        if vp:
            return {
                "poc": vp.poc,
                "vah": vp.vah,
                "val": vp.val,
                "price": vp.current_price,
                "price_vs_poc": "above" if vp.current_price > vp.poc else "below" if vp.current_price < vp.poc else "at",
                "delta_ratio": vp.delta_ratio,
                "imbalance": vp.imbalance.value,
            }
    except Exception:
        pass
    return None


# ─── Day trade momentum signal ────────────────────────────────────────

def daytrade_signal_15m(df: pd.DataFrame) -> Dict[str, Any]:
    """Tín hiệu intraday cho khung 15m/5m."""
    if df.empty or len(df) < 20:
        return {"trend": "WAIT", "score": 0, "details": []}

    close = df["Close"]
    last = df.iloc[-1]
    score = 0.0
    details: List[str] = []

    # 1. EMA ribbon (9/21/50)
    ema9 = float(close.ewm(span=9, adjust=False).mean().iloc[-1])
    ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
    ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1]) if len(df) >= 50 else ema21
    price_now = float(last["Close"])

    if price_now > ema9 > ema21 > ema50:
        score += 2
        details.append(f"EMA bullish ribbon (P>{ema9:.1f}>{ema21:.1f}>{ema50:.1f}) [+2]")
    elif price_now < ema9 < ema21 < ema50:
        score -= 2
        details.append(f"EMA bearish ribbon [-2]")
    elif price_now > ema9:
        score += 0.5
        details.append(f"Price > EMA9 [+0.5]")
    elif price_now < ema9:
        score -= 0.5
        details.append(f"Price < EMA9 [-0.5]")

    # 2. RSI(14) momentum
    rsi = float(last.get("RSI_14", np.nan)) if not pd.isna(last.get("RSI_14", np.nan)) else 50
    if rsi > 60:
        score += 0.5
        details.append(f"RSI14={rsi:.1f} bullish zone [+0.5]")
    elif rsi < 40:
        score -= 0.5
        details.append(f"RSI14={rsi:.1f} bearish zone [-0.5]")
    else:
        details.append(f"RSI14={rsi:.1f} neutral")

    # 3. MACD momentum
    macd = float(last.get("MACD", 0)) if not pd.isna(last.get("MACD", np.nan)) else 0
    macd_sig = float(last.get("MACD_Signal", 0)) if not pd.isna(last.get("MACD_Signal", np.nan)) else 0
    macd_hist = float(last.get("MACD_Hist", 0)) if not pd.isna(last.get("MACD_Hist", np.nan)) else 0

    if macd > macd_sig and macd_hist > 0:
        score += 1
        details.append(f"MACD bullish + hist rising [+1]")
    elif macd < macd_sig and macd_hist < 0:
        score -= 1
        details.append(f"MACD bearish + hist falling [-1]")
    elif macd > macd_sig:
        score += 0.5
        details.append(f"MACD > Signal [+0.5]")
    else:
        score -= 0.5
        details.append(f"MACD < Signal [-0.5]")

    # 4. Bollinger squeeze/expansion
    bb_width = float(last.get("BB_Width", 0)) if not pd.isna(last.get("BB_Width", np.nan)) else 0
    if bb_width > 0:
        bb_width_prev = float(df["BB_Width"].iloc[-5]) if len(df) >= 5 and not pd.isna(df["BB_Width"].iloc[-5]) else bb_width
        if bb_width > bb_width_prev * 1.3:
            details.append(f"BB expanding (volatility ↑) [momentum play]")

    # 5. Candlestick pattern: bullish/bearish engulfing
    if len(df) >= 2:
        prev_c = df.iloc[-2]
        prev_body = prev_c["Close"] - prev_c["Open"]
        curr_body = last["Close"] - last["Open"]

        if prev_body < 0 and curr_body > abs(prev_body) * 1.2:
            score += 1
            details.append("Bullish engulfing pattern [+1]")
        elif prev_body > 0 and abs(curr_body) > prev_body * 1.2 and curr_body < 0:
            score -= 1
            details.append("Bearish engulfing pattern [-1]")

    # 6. Volume spike
    vol = float(last.get("Volume", 0)) if not pd.isna(last.get("Volume", np.nan)) else 0
    vol_sma = float(df["Volume"].rolling(20).mean().iloc[-1]) if not pd.isna(df["Volume"].rolling(20).mean().iloc[-1]) else vol
    if vol_sma > 0 and vol > vol_sma * 1.5:
        if score > 0:
            score += 0.5
            details.append(f"Volume spike x{vol/vol_sma:.1f} confirming [+0.5]")
        elif score < 0:
            score -= 0.5
            details.append(f"Volume spike x{vol/vol_sma:.1f} confirming [-0.5]")

    trend = "UP" if score >= 1.5 else "DOWN" if score <= -1.5 else "SIDEWAYS"

    return {
        "trend": trend,
        "score": round(score, 1),
        "ema9": round(ema9, 2),
        "ema21": round(ema21, 2),
        "rsi": round(rsi, 1),
        "details": details,
    }


# ─── Day trade zones ──────────────────────────────────────────────────

def daytrade_zones(
    tf_data: Dict[str, pd.DataFrame],
    bias: str,
    decimals: int,
) -> Optional[Dict[str, Any]]:
    """Tính entry/exit zones cho day trade."""
    df_15m = tf_data.get("15m")
    df_1h = tf_data.get("1H")

    if df_15m is None or df_15m.empty:
        return None

    last_15m = df_15m.iloc[-1]
    price = float(last_15m["Close"])

    # ATR 14 trên 15m
    atr_15m = float(last_15m.get("ATR", np.nan)) if not pd.isna(last_15m.get("ATR", np.nan)) else float(price * 0.0015)

    # VWAP
    vwap = calculate_vwap(df_15m)

    # Volume Profile POC
    vp = volume_profile_poc(df_1h) if df_1h is not None and not df_1h.empty else None

    # 1H swing levels
    h1_df = tf_data.get("1H")
    h1_high_20 = float(h1_df["High"].iloc[-20:].max()) if h1_df is not None and len(h1_df) >= 20 else price * 1.01
    h1_low_20 = float(h1_df["Low"].iloc[-20:].min()) if h1_df is not None and len(h1_df) >= 20 else price * 0.99

    if "BUY" in bias:
        # Entry near support: VWAP hoặc VAL hoặc swing low 20
        supports = []
        if vwap is not None and vwap < price:
            supports.append(vwap)
        if vp is not None and vp["val"] < price:
            supports.append(vp["val"])
        if h1_low_20 < price:
            supports.append(h1_low_20)

        entry_low = max(supports) if supports else price - atr_15m
        entry_high = price
        sl = round(entry_low - atr_15m * DAY_ATR_SL, decimals)
        tp1 = round(price + atr_15m * DAY_ATR_TP1, decimals)
        tp2 = round(price + atr_15m * DAY_ATR_TP2, decimals)

        # Adjust TP to nearest resistance (only if closer than tp2 but better than tp1)
        if h1_high_20 > price and tp1 < h1_high_20 < tp2:
            tp2 = round(h1_high_20, decimals)

        risk = price - sl
        rr1 = round((tp1 - price) / risk, 1) if risk > 0 else 0
        rr2 = round((tp2 - price) / risk, 1) if risk > 0 else 0

        key_level = f"VWAP:{fmt_price(vwap, decimals)}" if vwap else f"1H Low20:{fmt_price(h1_low_20, decimals)}"
    else:
        resistances = []
        if vwap is not None and vwap > price:
            resistances.append(vwap)
        if vp is not None and vp["vah"] > price:
            resistances.append(vp["vah"])
        if h1_high_20 > price:
            resistances.append(h1_high_20)

        entry_high = min(resistances) if resistances else price + atr_15m
        entry_low = price
        sl = round(entry_high + atr_15m * DAY_ATR_SL, decimals)
        tp1 = round(price - atr_15m * DAY_ATR_TP1, decimals)
        tp2 = round(price - atr_15m * DAY_ATR_TP2, decimals)

        if h1_low_20 < price and tp2 < h1_low_20 < tp1:
            tp2 = round(h1_low_20, decimals)

        risk = sl - price
        rr1 = round((price - tp1) / risk, 1) if risk > 0 else 0
        rr2 = round((price - tp2) / risk, 1) if risk > 0 else 0

        key_level = f"VWAP:{fmt_price(vwap, decimals)}" if vwap else f"1H High20:{fmt_price(h1_high_20, decimals)}"

    return {
        "entry_zone": f"{fmt_price(entry_low, decimals)} - {fmt_price(entry_high, decimals)}",
        "stop_loss": fmt_price(sl, decimals),
        "tp1": fmt_price(tp1, decimals),
        "tp2": fmt_price(tp2, decimals),
        "rr1": rr1,
        "rr2": rr2,
        "atr_15m": fmt_price(atr_15m, decimals),
        "key_level": key_level,
    }


# ─── Build day trade report ───────────────────────────────────────────

def build_daytrade_report(tf_data: Dict[str, pd.DataFrame], cfg: Dict[str, Any]) -> str:
    d = cfg["decimals"]
    name = cfg["display_name"]

    for tf in ["4H", "1H", "15m"]:
        if tf not in tf_data or tf_data[tf].empty:
            return f"ERROR: Need {tf} data for day trading."

    # Bias từ 4H
    h4_trend, h4_score = determine_trend(tf_data["4H"])

    # Structure từ 1H
    h1_trend, h1_score = determine_trend(tf_data["1H"])

    # Entry signal từ 15m
    sig_15m = daytrade_signal_15m(tf_data["15m"])
    sig_5m = daytrade_signal_15m(tf_data.get("5m", tf_data["15m"]))

    # Tổng score
    dt_score = (
        (1 if h4_trend == "UP" else -1 if h4_trend == "DOWN" else 0) * 2.0
        + (1 if h1_trend == "UP" else -1 if h1_trend == "DOWN" else 0) * 1.5
        + sig_15m["score"] * 1.0
        + sig_5m["score"] * 0.5
    )

    if dt_score >= 4:
        dt_bias = "STRONG BUY"
        bias_icon = "🟢"
    elif dt_score >= 2:
        dt_bias = "BUY"
        bias_icon = "🟢"
    elif dt_score <= -4:
        dt_bias = "STRONG SELL"
        bias_icon = "🔴"
    elif dt_score <= -2:
        dt_bias = "SELL"
        bias_icon = "🔴"
    else:
        dt_bias = "WAIT"
        bias_icon = "🟡"

    # VWAP & Volume Profile
    vwap = calculate_vwap(tf_data["15m"])
    vp = volume_profile_poc(tf_data["1H"])
    or_levels = opening_range_levels(tf_data["15m"])

    # Zones
    zones = daytrade_zones(tf_data, dt_bias, d) if dt_bias != "WAIT" else None

    # Build
    lines: List[str] = []
    lines.append("=" * 64)
    lines.append(f"  💹 DAY TRADING ANALYSIS — {name}")
    lines.append(f"  Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"  Style: Intraday (close all positions EOD)")
    lines.append("=" * 64)
    lines.append("")

    # Bias overview
    lines.append("【INTRADAY BIAS】")
    lines.append(f"  4H Bias:  {h4_trend} (score: {h4_score:+d}) — {'Theo LONG' if h4_trend=='UP' else 'Theo SHORT' if h4_trend=='DOWN' else 'Chờ breakout'}")
    lines.append(f"  1H Struct: {h1_trend} (score: {h1_score:+d})")
    lines.append("")

    # Intraday levels
    lines.append("【INTRADAY LEVELS】")
    if vwap:
        last_price = float(tf_data["15m"]["Close"].iloc[-1])
        vwap_pos = "ABOVE VWAP ▲" if last_price > vwap else "BELOW VWAP ▼"
        lines.append(f"  VWAP: {fmt_price(vwap, d)} — Price is {vwap_pos}")
    else:
        lines.append(f"  VWAP: N/A")

    if vp:
        lines.append(f"  Volume Profile POC: {fmt_price(vp['poc'], d)}")
        lines.append(f"  Value Area: {fmt_price(vp['val'], d)} – {fmt_price(vp['vah'], d)}")
        lines.append(f"  Price vs POC: {vp['price_vs_poc']}")

    if or_levels:
        lines.append(f"  Opening Range: {fmt_price(or_levels['or_low'], d)} – {fmt_price(or_levels['or_high'], d)} "
                     f"({fmt_price(or_levels['or_range'], d)}) — Price: {or_levels['position']}")
    lines.append("")

    # 15m Signal
    lines.append(f"【15M ENTRY SIGNAL】Score: {sig_15m['score']:+.1f} | Trend: {sig_15m['trend']}")
    lines.append(f"  EMA9: {fmt_price(sig_15m['ema9'], d)} | EMA21: {fmt_price(sig_15m['ema21'], d)} | RSI14: {sig_15m['rsi']}")
    for detail in sig_15m["details"]:
        lines.append(f"  - {detail}")
    lines.append("")

    # 5m Precision
    lines.append(f"【5M PRECISION】Score: {sig_5m['score']:+.1f} | Trend: {sig_5m['trend']}")
    for detail in sig_5m["details"]:
        lines.append(f"  - {detail}")
    lines.append("")

    # Overall
    lines.append(f"【DAY TRADE SIGNAL: {bias_icon} {dt_bias} (score: {dt_score:+.1f})】")

    if zones:
        lines.append("")
        lines.append("【ENTRY & EXECUTION】")
        lines.append(f"  Entry Zone:    {zones['entry_zone']}")
        lines.append(f"  Stop Loss:     {zones['stop_loss']}")
        lines.append(f"  Take Profit 1: {zones['tp1']} (R:R = 1:{zones['rr1']})")
        lines.append(f"  Take Profit 2: {zones['tp2']} (R:R = 1:{zones['rr2']})")
        lines.append(f"  ATR(15m):      {zones['atr_15m']}")
        lines.append(f"  Key Level:     {zones['key_level']}")
        lines.append("")
        lines.append("  📋 Day Trading Rules:")
        lines.append("  • Entry: Limit order tại vùng giá, xác nhận bằng 5m engulfing/pinbar")
        lines.append("  • SL: Tuyệt đối, không dời xa hơn. Đặt sau key level + 1 ATR")
        lines.append("  • TP1: Đóng 50%, dời SL về entry")
        lines.append("  • TP2: Đóng 25%, trail 25% còn lại với EMA9(15m)")
        lines.append("  • Max 3 trades/ngày. Nếu -2% daily, dừng giao dịch")
        lines.append("  • Tránh giao dịch 30 phút trước/sau tin tức quan trọng")
        lines.append("  • Thời điểm tốt nhất: London Open (14:00-17:00 GMT+7), NY Open (19:30-22:30)")
    else:
        lines.append("")
        lines.append("  ⏸️  No clear intraday setup. Đợi:")
        lines.append("  • Break và retest của Opening Range")
        lines.append("  • Pullback về VWAP trong xu hướng")
        lines.append("  • Volume spike xác nhận breakout")

    lines.append("")
    lines.append("=" * 64)
    return "\n".join(lines)


# ─── Quick signal ─────────────────────────────────────────────────────

def build_daytrade_summary(tf_data: Dict[str, pd.DataFrame], cfg: Dict[str, Any]) -> str:
    d = cfg["decimals"]
    h4_trend, _ = determine_trend(tf_data.get("4H", pd.DataFrame()))
    h1_trend, _ = determine_trend(tf_data.get("1H", pd.DataFrame()))
    sig_15m = daytrade_signal_15m(tf_data.get("15m", pd.DataFrame()))

    dt_score = (
        (1 if h4_trend == "UP" else -1 if h4_trend == "DOWN" else 0) * 2.0
        + (1 if h1_trend == "UP" else -1 if h1_trend == "DOWN" else 0) * 1.5
        + sig_15m["score"] * 1.0
    )

    if dt_score >= 2:
        bias = "BUY"
    elif dt_score <= -2:
        bias = "SELL"
    else:
        bias = "WAIT"

    return (
        f"DAY[{cfg['display_name']}] 4H:{h4_trend} 1H:{h1_trend} "
        f"15m:{sig_15m['trend']}({sig_15m['score']:+.1f}) → {bias} ({dt_score:+.1f})"
    )


# ─── Entry point ──────────────────────────────────────────────────────

def run_daytrade_analysis(instrument: str = "xau", no_cache: bool = False) -> str:
    if instrument not in INSTRUMENTS:
        return f"Unknown: {instrument}"

    cfg = INSTRUMENTS[instrument]
    tf_data = fetch_all_timeframes(cfg, use_cache=not no_cache)

    if not tf_data:
        return "No data."

    for tf in ["4H", "1H", "15m"]:
        if tf not in tf_data or tf_data[tf].empty:
            return f"ERROR: Missing {tf} data for day trade."

    return build_daytrade_report(tf_data, cfg)


def run_daytrade_signal(instrument: str = "xau", no_cache: bool = False) -> str:
    if instrument not in INSTRUMENTS:
        return f"Unknown: {instrument}"

    cfg = INSTRUMENTS[instrument]
    tf_data = fetch_all_timeframes(cfg, use_cache=not no_cache)

    if not tf_data:
        return "No data."

    return build_daytrade_summary(tf_data, cfg)
